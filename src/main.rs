use std::collections::VecDeque;
use std::{error::Error, fs::File};

#[derive(Debug, Clone)]
struct Row {
    ticker: String,
    _index: usize,
    price: f64,
    is_minima: f64,
}

fn get_data(path: String) -> Result<Vec<Row>, Box<dyn Error>> {
    let file = File::open(path)?;
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

    pre_process_data(&mut records);

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

fn forward_pass(
    input_vector: Vec<f64>,
    hidden_layer_weights: Vec<Vec<f64>>,
    hidden_layer_biases: Vec<f64>,
    output_weights: Vec<f64>,
    output_bias: f64,
) -> (f64, Vec<f64>) {
    // Hidden layers internal score
    let mut neuron_signals: Vec<f64> = Vec::new();
    for (neuron_weights, bias) in hidden_layer_weights.iter().zip(hidden_layer_biases.iter()) {
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
    let mut output_signal: f64 = 0f64;
    for (neuron_signal, output_weight) in neuron_signals.iter().zip(output_weights.iter()) {
        output_signal += neuron_signal * output_weight;
    }
    output_signal += output_bias;

    // Output neuron activation
    output_signal = sigmoid(output_signal);

    (output_signal, neuron_signals)
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

    let weight = if label == 1.0 { 10.0 } else { 1.0 }; // Due to infrequent 1's
    let gradient = (prediction_propability - label) * weight;

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

fn train(
    window_size: usize,
    hidden_layer_weights: &mut Vec<Vec<f64>>,
    hidden_layer_biases: &mut Vec<f64>,
    output_weights: &mut Vec<f64>,
    output_bias: &mut f64,
) {
    let learning_rate: f64 = 0.02;
    let epochs = 20;

    let records: Vec<Row> = get_data("training_data.csv".to_string()).unwrap();

    let mut input_vector: VecDeque<f64> = VecDeque::new();
    let mut label: f64 = 0.0;
    let mut prev_ticker = "";

    for epoch in 0..epochs {
        for i in window_size..records.len() {
            if records[i].ticker != prev_ticker {
                input_vector.clear();
                for j in 1..records.len() {
                    input_vector.push_back(records[j].price);

                    if j == window_size {
                        label = records[j].is_minima;
                        break;
                    }
                }
                prev_ticker = &records[i].ticker;
            }

            let (prediction_probability, hidden_activations) = forward_pass(
                Vec::from(input_vector.clone()),
                hidden_layer_weights.clone(),
                hidden_layer_biases.clone(),
                output_weights.clone(),
                *output_bias,
            );

            let loss = binary_cross_entropy_loss(prediction_probability, label);
            println!("{:?}", loss);

            backward_pass(
                learning_rate,
                label,
                Vec::from(input_vector.clone()),
                prediction_probability,
                hidden_activations,
                hidden_layer_weights,
                hidden_layer_biases,
                output_weights,
                output_bias,
            );

            input_vector.pop_front();
            input_vector.push_back(records[i].price);
            label = records[i].is_minima;
        }
    }
}

fn predict(
    input_vector: Vec<f64>,
    hidden_layer_weights: &mut Vec<Vec<f64>>,
    hidden_layer_biases: &mut Vec<f64>,
    output_weights: &mut Vec<f64>,
    output_bias: &mut f64,
) -> f64 {
    let (prediction_probability, _) = forward_pass(
        input_vector.clone(),
        hidden_layer_weights.clone(),
        hidden_layer_biases.clone(),
        output_weights.clone(),
        *output_bias,
    );

    prediction_probability
}

fn evaluate(
    window_size: usize,
    hidden_layer_weights: &mut Vec<Vec<f64>>,
    hidden_layer_biases: &mut Vec<f64>,
    output_weights: &mut Vec<f64>,
    output_bias: &mut f64,
) {
}

fn main() {
    // TODO: Randomize
    let mut hidden_layer_weights: Vec<Vec<f64>> = vec![vec![0.1f64; 100]; 50];
    let mut hidden_layer_biases: Vec<f64> = vec![0f64; 50];

    let mut output_weights: Vec<f64> = vec![0.1f64; 50];
    let mut output_bias: f64 = 0f64;

    let window_size: usize = 100;

    train(
        window_size.clone(),
        &mut hidden_layer_weights,
        &mut hidden_layer_biases,
        &mut output_weights,
        &mut output_bias,
    );

    /*
    evaluate(
        window_size,
        &mut hidden_layer_weights,
        &mut hidden_layer_biases,
        &mut output_weights,
        &mut output_bias,
    );
    */
}
