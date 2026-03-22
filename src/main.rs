use std::{error::Error, fs::File};

#[derive(Debug, Clone)]
struct Row {
    ticker: String,
    _index: usize,
    price: f64,
    is_minima: f64,
}

fn get_data() -> Result<Vec<Row>, Box<dyn Error>> {
    let file = File::open("training_data.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut records: Vec<Row> = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let row = Row {
            ticker: record[0].to_string(),
            _index: record[1].parse::<usize>()?,
            price: record[2].parse::<f64>()?,
            is_minima: record[3].parse::<f64>()?,
        };

        records.push(row.clone());
    }

    Ok(records)
}

fn pre_process_data(records: &mut Vec<Row>) {
    let mut prev_price = records[0].price;

    // Normalize price
    for record in records.iter_mut().skip(1) {
        let new_price = (record.price - prev_price) / prev_price;
        prev_price = record.price;
        record.price = new_price;
    }

    records.remove(0); // Can't take the difference when there is no previous price.
}

/*
// Xavier initialization (common for sigmoid)
let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
w = random_uniform(-limit, limit)

// He initialization (common for ReLU)
let std = (2.0 / fan_in as f64).sqrt();
w = random_normal(0.0, std)
*/

// Forward pass math:
// y_scalar = sigmoid( output_weights * ReLU( hidden_weight * input_vector + hidden_biases ) + output_biases )

fn init_params() {}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn forward_pass(input_vector: Vec<f64>) -> (f64, Vec<f64>, f64, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    /*
     * hidden_weights size: hidden_layer_neurons_count * input_dimensions,
     * 50 * 100,
     * 50 rows, 100 columns,
     * there is 50 neuron and each neuron has 100 inputs,
     */
    let hidden_weights: Vec<Vec<f64>> = vec![vec![0.1f64; 100]; 50];
    let hidden_biases: Vec<f64> = vec![0f64; 50];

    // Hidden layers internal score
    let mut neuron_signals: Vec<f64> = Vec::new();
    for (neuron_weights, bias) in hidden_weights.iter().zip(hidden_biases.iter()) {
        let mut sum = 0f64;

        for (weight, input) in neuron_weights.iter().zip(input_vector.iter()) {
            sum += weight * input;
        }

        sum += bias;
        neuron_signals.push(sum);
    }

    // Hidden layers activation
    for score in neuron_signals.iter_mut() {
        *score = relu(*score);
    }

    // Output neuron internal score
    let output_weights: Vec<f64> = vec![0.1f64; 50];
    let output_bias: f64 = 0f64;

    let mut output_signal: f64 = 0f64;
    for (neuron_signal, output_weight) in neuron_signals.iter().zip(output_weights.iter()) {
        output_signal += neuron_signal * output_weight;
    }
    output_signal += output_bias;

    // Output neuron activation
    output_signal = sigmoid(output_signal);

    (
        output_signal,
        output_weights,
        output_bias,
        neuron_signals,
        hidden_weights,
        hidden_biases,
    )
}

fn binary_cross_entropy_loss(prediction_propability: f64, label: f64) -> f64 {
    let epsilon: f64 = 1e-15;
    let propability: f64 = prediction_propability.clamp(epsilon, 1.0 - epsilon);

    -(label * propability.ln() + (1.0 - label) * (1.0 - propability).ln())
}

fn backward_pass(
    learning_rate: f64,
    label: f64,
    inputs: Vec<f64>,
    prediction_propability: f64,
    hidden_layer_activations: Vec<f64>,
    hidden_layer_weights: &mut Vec<Vec<f64>>,
    hidden_layer_biases: &mut Vec<f64>,
    output_weights: &mut Vec<f64>,
    output_bias: &mut f64,
) {
    // Correction for output layer:
    // Output_Neuron_Weight_i -= Output_Neuron_Activation_i * (prediction - label) * learning_rate

    let original_output_weights = output_weights.clone();

    let gradient = prediction_propability - label;

    for (weight, activation) in output_weights
        .iter_mut()
        .zip(hidden_layer_activations.iter())
    {
        *weight -= learning_rate * gradient * activation;
    }
    *output_bias -= learning_rate * gradient;

    // Correction for hidden layer:
    // Hidden_Layer_Neuron_Weight_i_j -= input[j] * ((prediction - label) * Output_Neuron_Weight_i
    // * ReLU_Derivative) * learning_rate

    for i in 0..hidden_layer_activations.len() {
        let relu_derivative: f64 = if hidden_layer_activations[i] > 0.0 {
            1.0
        } else {
            0.0
        };

        let hidden_gradient = gradient * original_output_weights[i] * relu_derivative;

        for j in 0..hidden_layer_weights[i].len() {
            hidden_layer_weights[i][j] -= inputs[j] * hidden_gradient * learning_rate;
        }
        hidden_layer_biases[i] -= learning_rate * hidden_gradient;
    }
}

fn train() {}

fn predict() {}

fn evaluate() {}

fn main() {
    let mut records: Vec<Row> = get_data().unwrap();
    pre_process_data(&mut records);

    let mut input_vector: Vec<f64> = Vec::new();

    for index in 91..records.len() {
        input_vector.push(records[index].price);

        if index == 190 {
            break;
        }
    }
    let label: f64 = records[190].is_minima;

    println!("{:?}", input_vector.len());
    println!("{:?}", label);

    let (
        prediction_propability,
        mut output_weights,
        mut output_bias,
        hidden_activations,
        mut hidden_weights,
        mut hidden_biases,
    ) = forward_pass(input_vector.clone());
    println!("{:?}", prediction_propability);

    let loss = binary_cross_entropy_loss(prediction_propability, label);
    println!("{:?}", loss);

    let learning_rate = 0.02;

    backward_pass(
        learning_rate,
        label,
        input_vector.clone(),
        prediction_propability,
        hidden_activations,
        &mut hidden_weights,
        &mut hidden_biases,
        &mut output_weights,
        &mut output_bias,
    );

    let (
        prediction_propability,
        mut output_weights,
        mut output_bias,
        hidden_activations,
        mut hidden_weights,
        mut hidden_biases,
    ) = forward_pass(input_vector.clone());
    println!("{:?}", prediction_propability);

    let loss = binary_cross_entropy_loss(prediction_propability, label);
    println!("{:?}", loss);
}
