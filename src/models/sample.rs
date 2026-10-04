use luma::{Device, Tensor};
use rand::RngExt;
use std::cmp::Ordering;

pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
}

impl Sampler {
    pub fn new(temperature: f64, top_p: f64, top_k: usize) -> Self {
        Self { temperature, top_p, top_k }
    }

    /// 1. 贪婪/精确配置 (等价于 GreedySearch)
    /// 适用于：代码生成、数学计算、JSON 格式化提取
    pub fn strict() -> Self {
        Self::new(0.1, 1.0, 1)
    }

    /// 2. 均衡/聊天配置 (主流默认值)
    /// 适用于：通用问答助理、闲聊、文章总结
    pub fn chat_default() -> Self {
        Self::new(0.7, 0.95, 50)
    }

    /// 3. 创造力配置
    /// 适用于：小说续写、头脑风暴、角色扮演
    pub fn creative() -> Self {
        Self::new(1.1, 0.99, 100)
    }

    /// 4. 严谨分析配置
    /// 适用于：逻辑推理、事实问答、数据分析
    pub fn reasoning() -> Self {
        Self::new(0.4, 0.85, 20)
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Sample one token per row of `logits` (`(num_seqs, vocab)`).
    ///
    /// `temperatures` is per sequence; a value `<= 0` means greedy. When fewer
    /// temperatures than rows are given, the sampler's default is used.
    pub fn sample<Dev: Device>(&self, logits: &Tensor<Dev>, temperatures: &[f64]) -> anyhow::Result<Vec<u32>> {
        let dims = logits.dims();
        anyhow::ensure!(!dims.is_empty(), "sample: empty logits shape");
        let vocab = *dims.last().unwrap();
        let flat = logits.to_vec()?;
        anyhow::ensure!(vocab > 0, "sample: vocab must be positive");
        let rows = flat.len() / vocab;

        let mut tokens = Vec::with_capacity(rows);
        for row in 0..rows {
            let logits = &flat[row * vocab..(row + 1) * vocab];
            let temperature = temperatures.get(row).copied().unwrap_or(self.temperature);
            tokens.push(self.sample_row(logits, temperature));
        }
        Ok(tokens)
    }

    fn sample_row(&self, logits: &[f64], temperature: f64) -> u32 {
        // temperature <= 0: greedy
        if temperature <= 0.0 {
            return self.argmax(logits);
        }

        // temperature scaling
        let scaled: Vec<f64> = logits.iter().map(|x| x / temperature).collect();
        let probs = self.softmax(&scaled);
        let filtered_probs = self.apply_top_k_top_p(probs);
        self.multinomial_sample(&filtered_probs)
    }

    fn argmax(&self, logits: &[f64]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap() as u32
    }

    fn softmax(&self, x: &[f64]) -> Vec<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&v| v / sum).collect()
    }

    fn apply_top_k_top_p(&self, probs: Vec<f64>) -> Vec<(u32, f64)> {
        let mut sorted_probs: Vec<(u32, f64)> = probs.into_iter().enumerate().map(|(i, v)| (i as u32, v)).collect();
        sorted_probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // top-k
        if self.top_k > 0 && self.top_k < sorted_probs.len() {
            sorted_probs.truncate(self.top_k);
        }

        // top-p
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff_index = sorted_probs.len();

            for (i, (_, prob)) in sorted_probs.iter().enumerate() {
                cumsum += prob;
                if cumsum > self.top_p {
                    cutoff_index = i + 1;
                    break;
                }
            }

            sorted_probs.truncate(cutoff_index);
        }

        sorted_probs
    }

    fn multinomial_sample(&self, sorted_probs: &[(u32, f64)]) -> u32 {
        let total_prob: f64 = sorted_probs.iter().map(|(_, p)| p).sum();
        let mut rng = rand::rng();
        let mut rng_val = rng.random_range(0.0..1.0) * total_prob;

        for (index, prob) in sorted_probs {
            rng_val -= prob;
            if rng_val <= 0.0 {
                return *index;
            }
        }

        sorted_probs.last().map(|(index, _)| *index).unwrap_or(0)
    }
}
