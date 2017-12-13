use std::ops::Deref;
use std::fmt::Debug;
use ndarray::{Array1, Array2};
use rand::Rng;
use activation::ActivationPolicy;

#[derive(Debug)]
pub struct Network<A: ActivationPolicy> {
    layers: Vec<Layer<A>>
}

pub struct Layer<A: ActivationPolicy> {
    n_nodes: usize,
    n_output_nodes: usize,

    // n_nodes * n_output_nodes
    weights: Array2<f32>,

    // n_output_nodes
    biases: Array1<f32>,

    activation_policy: A
}

pub struct LayerConfig<A: ActivationPolicy> {
    pub n_nodes: usize,
    pub n_output_nodes: usize,
    pub activation_policy: A
}

impl<A> Layer<A> where A: ActivationPolicy {
    pub fn new<T>(config: LayerConfig<T>) -> Layer<T> where T: ActivationPolicy {
        let mut weights = Array2::<f32>::zeros((config.n_output_nodes, config.n_nodes));
        let mut biases = Array1::<f32>::zeros((config.n_output_nodes,));
        let mut rng = ::rand::thread_rng();

        for i in 0..config.n_output_nodes {
            for j in 0..config.n_nodes {
                let ::rand::Open01(v) = rng.gen::<::rand::Open01<f32>>();
                weights[(i, j)] = v;
            }
        }

        for i in 0..config.n_output_nodes {
            let ::rand::Open01(v) = rng.gen::<::rand::Open01<f32>>();
            biases[i] = v;
        }

        Layer {
            n_nodes: config.n_nodes,
            n_output_nodes: config.n_output_nodes,
            weights: weights,
            biases: biases,
            activation_policy: config.activation_policy
        }
    }

    pub fn evaluate(&self, input: &Array1<f32>) -> Array1<f32> {
        let input_shape = input.shape();
        if input_shape[0] != self.n_nodes {
            panic!("Invalid input shape")
        }

        let mut output = self.weights.dot(input);
        assert!(output.shape()[0] == self.n_output_nodes);

        for i in 0..self.n_output_nodes {
            output[i] = self.activation_policy.activate(output[i] - self.biases[i]);
        }

        output
    }
}

impl<A> Debug for Layer<A> where A: ActivationPolicy {
    fn fmt(&self, formatter: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(
            formatter,
            "Layer {{ n_nodes: {:?}, n_output_nodes: {:?}, weights: {:?}, biases: {:?} }}",
            self.n_nodes, self.n_output_nodes, self.weights, self.biases
        )
    }
}

struct TrainingLayer<'a, A: ActivationPolicy + 'a> {
    inner: &'a Layer<A>,
    deltas: Array1<f32>
}

impl<'a, A> Deref for TrainingLayer<'a, A> where A: ActivationPolicy {
    type Target = Layer<A>;
    fn deref(&self) -> &Layer<A> {
        self.inner
    }
}

impl<A> Network<A> where A: ActivationPolicy {
    pub fn new() -> Network<A> {
        Network {
            layers: Vec::new()
        }
    }

    pub fn add_layer(&mut self, layer: Layer<A>) {
        self.layers.push(layer);
    }

    pub fn evaluate(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut current = input.clone();

        for layer in self.layers.iter() {
            current = layer.evaluate(&current);
        }

        current
    }

    pub fn evaluate_all(&self, input: &Array1<f32>) -> Vec<Array1<f32>> {
        let mut outputs: Vec<Array1<f32>> = Vec::new();

        for layer in self.layers.iter() {
            let v = layer.evaluate(outputs.iter().last().unwrap_or(input));
            outputs.push(v);
        }

        outputs
    }

    pub fn backpropagate(
        &mut self,
        input: &Array1<f32>,
        expected_output: &Array1<f32>
    ) {
        let current_outputs = self.evaluate_all(input);
        let mut training_layers: Vec<TrainingLayer<A>> = self.layers.iter().map(|v| TrainingLayer {
            inner: v,
            deltas: Array1::<f32>::zeros((v.n_output_nodes,))
        }).collect();
    }
}

#[test]
fn test_forward_eval() {
    use activation::Relu;

    let input: Array1<f32> = Array1::from(vec! [ 1.0, 1.2, 3.5 ]);
    let mut network: Network<Relu> = Network::new();

    network.add_layer(Layer::<Relu>::new(LayerConfig {
        n_nodes: 3,
        n_output_nodes: 5,
        activation_policy: Relu::default()
    }));
    network.add_layer(Layer::<Relu>::new(LayerConfig {
        n_nodes: 5,
        n_output_nodes: 2,
        activation_policy: Relu::default()
    }));

    let output = network.evaluate_all(&input);
    println!("{:?}", output);
}
