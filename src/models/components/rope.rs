use anyhow::Context;
use luma::{D, Device, Float, FloatDType, IntDType, Tensor, nn::{Buffer, Module}, no_grad, tensor::{FloatTensor, IntTensor}};



#[derive(Module)]
pub struct Rope<Dev: Device> {
    pub sin: Buffer<Dev, Float>,
    pub cos: Buffer<Dev, Float>,
    #[module(skip)]
    pub max_position_embeddings: usize,
}

impl<Dev: Device> Rope<Dev> {
    pub fn new(head_size: usize, max_position_embeddings: usize, rope_theta: f64, dtype: FloatDType, device: &Dev) -> anyhow::Result<Self> {
        no_grad!();
        let freqs = (0..head_size).step_by(2).map(|i| 1.0 / (rope_theta.powf(i as f64 / head_size as f64))).collect::<Vec<_>>();
        let freqs = FloatTensor::<Dev>::from_slice(&freqs, (1, freqs.len()), (device, dtype)).context("rope freqs from_slice")?;
        let positions = IntTensor::<Dev>::arange(0, max_position_embeddings as i64, 1, (device, IntDType::U32))
            .context("rope positions arange")?;
        // (max_positions, 1)
        let positions = positions.cast(dtype).context("rope positions cast")?.unsqueeze(1).context("rope positions unsqueeze")?;
        // (max_positions, head_size/2)
        let thetas = positions.matmul(&freqs).context("rope positions @ freqs")?;

        // (max_positions, head_size/2)
        let sin = thetas.sin().context("rope sin")?;
        let cos = thetas.cos().context("rope cos")?;

        // (max_positions, head_size)
        let sin = Tensor::cat(&[&sin, &sin], 1).context("rope cat sin")?;
        let cos = Tensor::cat(&[&cos, &cos], 1).context("rope cat cos")?;

        Ok(Self {
            sin: Buffer::<Dev, Float>::new(sin),
            cos: Buffer::<Dev, Float>::new(cos),
            max_position_embeddings,
        })
    }

    pub fn apply_rotary_emb(&self, x: &Tensor<Dev>, start_pos: usize) -> anyhow::Result<Tensor<Dev>> {
        let (_batch_size, _n_heads, seq_len, _head_size) = x.dims4().context("apply_rotary_emb dims4")?;
        assert!(start_pos + seq_len <= self.max_position_embeddings);
        let sin = self.sin.narrow(0, start_pos, seq_len).context("rope narrow sin")?; // (seq_len, head_size)
        let cos = self.cos.narrow(0, start_pos, seq_len).context("rope narrow cos")?; // (seq_len, head_size)
        let sin = sin.unsqueeze(0).context("rope sin unsqueeze batch")?.unsqueeze(0).context("rope sin unsqueeze head")?; // (1, 1, seq_len, head_size)
        let cos = cos.unsqueeze(0).context("rope cos unsqueeze batch")?.unsqueeze(0).context("rope cos unsqueeze head")?; // (1, 1, seq_len, head_size)

        // [x1, x2] -> [-x2, x1]
        let x_rotated = self.rotate_half(x).context("rotate_half")?;
        let x_out = x.broadcast_mul(&cos).context("rope x * cos")? + x_rotated.broadcast_mul(&sin).context("rope x_rotated * sin")?;
        Ok(x_out)
    }

    /// Per-token RoPE for the paged path.
    /// `x`: `(num_tokens, num_heads, head_dim)`, `positions`: `(num_tokens,)`.
    pub fn apply_rotary_emb_positions(&self, x: &Tensor<Dev>, positions: &IntTensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let (num_tokens, _n_heads, head_size) = x.dims3().context("apply_rotary_emb_positions dims3")?;
        let sin = self.sin.index_select(positions, 0).context("rope gather sin")?; // (num_tokens, head_size)
        let cos = self.cos.index_select(positions, 0).context("rope gather cos")?; // (num_tokens, head_size)
        let sin = sin.reshape((num_tokens, 1, head_size)).context("rope sin reshape")?;
        let cos = cos.reshape((num_tokens, 1, head_size)).context("rope cos reshape")?;

        let x_rotated = self.rotate_half_last(x).context("rotate_half_last")?;
        let x_out = x.broadcast_mul(&cos).context("rope x * cos")? + x_rotated.broadcast_mul(&sin).context("rope x_rotated * sin")?;
        Ok(x_out)
    }

    /// `[x1, x2] -> [-x2, x1]` along the last dim, for any rank.
    fn rotate_half_last(&self, x: &Tensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let head_size = *x.dims().last().context("rotate_half_last empty dims")?;
        let half_size = head_size / 2;
        let x1 = x.narrow(D::Minus1, 0, half_size).context("rotate_half_last narrow x1")?;
        let x2 = x.narrow(D::Minus1, half_size, half_size).context("rotate_half_last narrow x2")?;
        let neg_x2 = x2.neg().context("rotate_half_last neg x2")?;
        Ok(Tensor::cat(&[&neg_x2, &x1], D::Minus1).context("rotate_half_last cat")?)
    }

    fn rotate_half(&self, x: &Tensor<Dev>) -> anyhow::Result<Tensor<Dev>> {
        let (_batch_size, _n_heads, _seq_len, head_size) = x.dims4().context("rotate_half dims4")?;
        let half_size = head_size / 2;
        // (batch, n_heads, seq_len, head_size/2)
        let x1 = x.narrow(D::Minus1, 0, half_size).context("rotate_half narrow x1")?;
        let x2 = x.narrow(D::Minus1, half_size, half_size).context("rotate_half narrow x2")?;
        let neg_x2 = x2.neg().context("rotate_half neg x2")?;
        // (batch, n_heads, seq_len, head_size)
        let x_rotated = Tensor::cat(&[&neg_x2, &x1], D::Minus1).context("rotate_half cat")?;
        Ok(x_rotated)
    }
}