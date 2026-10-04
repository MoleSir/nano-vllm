use rand::RngExt;
use std::cmp::Ordering;

/// 一个请求的生成配置：采样旋钮 + 停止条件，并负责对单条 logits 采样。
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    /// 温度 / top-k / top-p（每个请求独立生效）
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    /// 最多生成多少个新 token
    pub max_new_tokens: usize,
    /// 是否忽略 eos、一直生成到 max_new_tokens
    pub ignore_eos: bool,
}

impl SamplingParams {
    /// 贪婪/精确配置 (等价于 GreedySearch)
    /// 适用于：代码生成、数学计算、JSON 格式化提取
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 1, top_p: 1.0, max_new_tokens: 64, ignore_eos: false }
    }

    /// 均衡/聊天配置 (主流默认值)
    /// 适用于：通用问答助理、闲聊、文章总结
    pub fn chat() -> Self {
        Self { temperature: 0.7, top_k: 50, top_p: 0.95, max_new_tokens: 64, ignore_eos: false }
    }

    /// 创造力配置
    /// 适用于：小说续写、头脑风暴、角色扮演
    pub fn creative() -> Self {
        Self { temperature: 1.1, top_k: 100, top_p: 0.99, max_new_tokens: 64, ignore_eos: false }
    }

    /// 严谨分析配置
    /// 适用于：逻辑推理、事实问答、数据分析
    pub fn reasoning() -> Self {
        Self { temperature: 0.4, top_k: 20, top_p: 0.85, max_new_tokens: 64, ignore_eos: false }
    }

    /// 对一条 logits（单个位置、长度为 vocab）采样一个 token。
    /// `temperature <= 0` 表示贪婪。
    pub fn sample(&self, logits: &[f64]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }

        let scaled: Vec<f64> = logits.iter().map(|x| x / self.temperature).collect();
        let probs = softmax(&scaled);
        let filtered = self.apply_top_k_top_p(probs);
        multinomial_sample(&filtered)
    }

    fn apply_top_k_top_p(&self, probs: Vec<f64>) -> Vec<(u32, f64)> {
        let mut sorted: Vec<(u32, f64)> = probs.into_iter().enumerate().map(|(i, v)| (i as u32, v)).collect();
        sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // top-k
        if self.top_k > 0 && self.top_k < sorted.len() {
            sorted.truncate(self.top_k);
        }

        // top-p
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff = sorted.len();
            for (i, (_, prob)) in sorted.iter().enumerate() {
                cumsum += prob;
                if cumsum > self.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            sorted.truncate(cutoff);
        }

        sorted
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::greedy()
    }
}

fn argmax(logits: &[f64]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap() as u32
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

fn multinomial_sample(sorted: &[(u32, f64)]) -> u32 {
    let total: f64 = sorted.iter().map(|(_, p)| p).sum();
    let mut rng = rand::rng();
    let mut r = rng.random_range(0.0..1.0) * total;

    for (index, prob) in sorted {
        r -= prob;
        if r <= 0.0 {
            return *index;
        }
    }

    sorted.last().map(|(index, _)| *index).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax() {
        let logits = [0.1, 0.2, 9.0, 0.3];
        assert_eq!(SamplingParams::greedy().sample(&logits), 2);
    }

    #[test]
    fn negative_temperature_is_greedy() {
        let logits = [0.1, 0.2, 9.0, 0.3];
        let p = SamplingParams { temperature: -1.0, top_k: 0, top_p: 1.0, max_new_tokens: 8, ignore_eos: false };
        assert_eq!(p.sample(&logits), 2);
    }

    #[test]
    fn top_k_limits_candidates() {
        // top-2 是 idx 3 与 idx 1
        let logits = [0.0, 5.0, 0.1, 9.5];
        let p = SamplingParams { temperature: 1.0, top_k: 2, top_p: 1.0, max_new_tokens: 8, ignore_eos: false };
        for _ in 0..1000 {
            let t = p.sample(&logits);
            assert!(t == 1 || t == 3, "top_k=2 只应返回 top-2，得到 {t}");
        }
    }

    #[test]
    fn top_p_keeps_only_the_head_mass() {
        // 概率几乎全在 idx 2，top_p=0.5 时只剩它
        let logits = [0.0, 0.0, 20.0, 0.0];
        let p = SamplingParams { temperature: 1.0, top_k: 0, top_p: 0.5, max_new_tokens: 8, ignore_eos: false };
        for _ in 0..100 {
            assert_eq!(p.sample(&logits), 2);
        }
    }
}
