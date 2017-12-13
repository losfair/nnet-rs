extern crate ndarray;
extern crate rand;

pub mod feedforward;
pub mod activation;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
