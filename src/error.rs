use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("空闲块不足:还需要 {needed} 个块,空闲队列里只有 {free} 个")]
pub struct OutOfBlocks {
    /// 这次分配还需要占用几个空闲块
    pub needed: usize,
    /// 空闲队列里还剩几个
    pub free: usize,
}

/// crate 统一错误类型。以后加别的错误(模型、采样、序列……)就往这里加变体。
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    OutOfBlocks(#[from] OutOfBlocks),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
