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
}

impl Sampler {
    pub fn sample<Dev: Device>(&self, logits: &Tensor<Dev>) -> anyhow::Result<u32> {
        let mut logits: Vec<_> = logits.to_vec()?;

        // temperature = 0, greedy sample
        if self.temperature <= 0.0 {
            return Ok(self.argmax(&logits));
        }

        // temerature scaling
        for x in &mut logits {
            *x /= self.temperature;
        }
        // softmax
        let probs = self.softmax(&logits);

        // top-k & top-p sampling
        let filtered_probs = self.apply_top_k_top_p(probs);

        // multinomial sampling
        Ok(self.multinomial_sample(&filtered_probs))
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

        unreachable!()
    }
}