use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::io::Write;
use std::{error::Error, fs::File};

const POS_WEIGHT: f64 = 15.0;
const THRESHOLD: f64 = 0.70;

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
    let mut i = 0;
    while i < records.len() {
        let ticker = records[i].ticker.clone();
        let end = records[i..]
            .iter()
            .position(|r| r.ticker != ticker)
            .map(|p| i + p)
            .unwrap_or(records.len());

        // Normalize within this ticker only
        let mut prev_price = records[i].price;
        for j in (i + 1)..end {
            let new_price = (records[j].price - prev_price) / prev_price;
            prev_price = records[j].price;
            records[j].price = new_price;
        }

        records.remove(i);
        i = end - 1;
    }
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

fn standardize(inputs: &mut Vec<f64>) {
    let mean = inputs.iter().sum::<f64>() / inputs.len() as f64;
    let std = (inputs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / inputs.len() as f64).sqrt();
    if std > 1e-8 {
        for x in inputs.iter_mut() {
            *x = (*x - mean) / std;
        }
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

fn binary_cross_entropy_loss(prediction_propability: f64, label: f64, is_training: bool) -> f64 {
    let epsilon: f64 = 1e-15;
    let p: f64 = prediction_propability.clamp(epsilon, 1.0 - epsilon);
    let pos_weight = if is_training { POS_WEIGHT } else { 1.0 };

    -(pos_weight * label * p.ln() + (1.0 - label) * (1.0 - p).ln())
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

    let gradient = if label == 1.0 {
        POS_WEIGHT * (prediction_propability - label)
    } else {
        prediction_propability - label
    };

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
            0.01
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
    let learning_rate: f64 = 0.01;
    //let epochs = 20;
    let epochs = 5;

    let records: Vec<Row> = get_data("training_data.csv".to_string()).unwrap();

    // Pre-compute all valid (input_vector, label) pairs
    let mut samples: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut i = 1;
    while i < records.len() {
        let start = i;
        let ticker = records[i].ticker.clone();
        while i < records.len() && records[i].ticker == ticker {
            i += 1;
        }
        let end = i;

        // For this ticker, build sliding windows
        for j in start..end {
            let window_end = j + window_size;
            if window_end >= end {
                break;
            }

            let input: Vec<f64> = (j..window_end).map(|k| records[k].price).collect();
            let label = records[window_end].is_minima;
            samples.push((input, label));
        }
    }

    let mut rng = rand::thread_rng();

    for epoch in 0..epochs {
        println!("epoch: {:?}/{:?}", epoch + 1, epochs);

        // Shuffle samples each epoch
        samples.shuffle(&mut rng);

        for (input_vector, label) in &samples {
            let mut input = input_vector.clone();
            standardize(&mut input);
            let (prediction_probability, hidden_activations) = forward_pass(
                input.clone(),
                hidden_layer_weights.clone(),
                hidden_layer_biases.clone(),
                output_weights.clone(),
                *output_bias,
            );
            //println!("Prediction: {:?}", prediction_probability);

            let loss = binary_cross_entropy_loss(prediction_probability, *label, true);
            //println!("Loss: {:?}", loss);

            backward_pass(
                learning_rate,
                *label,
                input.clone(),
                prediction_probability,
                hidden_activations,
                hidden_layer_weights,
                hidden_layer_biases,
                output_weights,
                output_bias,
            );
        }
    }
}

fn predict(
    mut input_vector: Vec<f64>,
    hidden_layer_weights: &mut Vec<Vec<f64>>,
    hidden_layer_biases: &mut Vec<f64>,
    output_weights: &mut Vec<f64>,
    output_bias: &mut f64,
) -> f64 {
    standardize(&mut input_vector);
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
    let records: Vec<Row> = get_data("evaluation_data.csv".to_string()).unwrap();

    let mut samples: Vec<(Vec<f64>, f64, usize, f64, String)> = Vec::new();

    // Make windows (inputs)
    let mut i = 1;
    while i < records.len() {
        let start = i;
        let ticker = records[i].ticker.clone();
        while i < records.len() && records[i].ticker == ticker {
            i += 1;
        }
        let end = i;

        for j in start..end {
            let window_end = j + window_size;
            if window_end >= end {
                break;
            }

            let input: Vec<f64> = (j..window_end).map(|k| records[k].price).collect();
            let label = records[window_end].is_minima;
            let index = window_end;
            let price = records[window_end].price;
            let ticker_clone = records[window_end].ticker.clone();
            samples.push((input, label, index, price, ticker_clone));
        }
    }

    let mut total_loss = 0.0;
    let mut correct = 0;
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    let mut true_negatives = 0;

    // Predict for each window and store statistics.
    for (input_vector, label, index, price, ticker) in &samples {
        let mut input = input_vector.clone();
        standardize(&mut input);
        let (prediction, _) = forward_pass(
            input,
            hidden_layer_weights.clone(),
            hidden_layer_biases.clone(),
            output_weights.clone(),
            *output_bias,
        );

        total_loss += binary_cross_entropy_loss(prediction, *label, false);

        let predicted_label = if prediction >= THRESHOLD { 1.0 } else { 0.0 };
        if predicted_label == *label {
            correct += 1;
        }
        if *label == 1.0 && predicted_label == 1.0 {
            true_positives += 1;
        }
        if *label == 0.0 && predicted_label == 1.0 {
            false_positives += 1;
        }
        if *label == 1.0 && predicted_label == 0.0 {
            false_negatives += 1;
        }
        if *label == 0.0 && predicted_label == 0.0 {
            true_negatives += 1;
        }
    }

    // Log results
    let num_samples = samples.len() as f64;
    println!("--- Evaluation ---");
    println!("Samples: {}", samples.len());
    println!("Avg Loss: {:.6}", total_loss / num_samples);
    println!(
        "Accuracy: {:.4} ({}/{})",
        correct as f64 / num_samples,
        correct,
        samples.len()
    );
    println!(
        "True Positives: {}  False Positives: {}  False Negatives: {}  True Negatives: {}",
        true_positives, false_positives, false_negatives, true_negatives
    );
    if true_positives + false_positives > 0 {
        println!(
            "Precision: {:.4}",
            true_positives as f64 / (true_positives + false_positives) as f64
        );
    }
    if true_positives + false_negatives > 0 {
        println!(
            "Recall: {:.4}",
            true_positives as f64 / (true_positives + false_negatives) as f64
        );
    }

    let mut file = File::create("predictions.csv").unwrap();

    // Save results for plotting.
    writeln!(file, "ticker,index,price,probability,label").unwrap();
    let mut total_loss = 0.0;

    for (input_vector, label, index, price, ticker) in &samples {
        let mut input = input_vector.clone();
        standardize(&mut input);

        let (prediction, _) = forward_pass(
            input,
            hidden_layer_weights.clone(),
            hidden_layer_biases.clone(),
            output_weights.clone(),
            *output_bias,
        );

        total_loss += binary_cross_entropy_loss(prediction, *label, false);

        writeln!(
            file,
            "{},{},{:.6},{:.6},{}",
            ticker, index, price, prediction, label
        )
        .unwrap();
    }

    println!("Saved predictions to predictions.csv");
}

fn init_params(window_size: usize, hidden_size: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, (2.0 / window_size as f64).sqrt()).unwrap();
    let xavier_limit = (6.0 / (hidden_size + 1) as f64).sqrt();

    // He initialization
    let hidden_weights = (0..hidden_size)
        .map(|_| (0..window_size).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    let hidden_biases = vec![0.0; hidden_size];

    // Xavier initialization
    let output_weights = (0..hidden_size)
        .map(|_| rng.gen_range(-xavier_limit..=xavier_limit))
        .collect();

    let output_bias = 0.0;

    (hidden_weights, hidden_biases, output_weights, output_bias)
}

fn main() {
    let window_size: usize = 100;
    let hidden_size: usize = 50;

    let (mut hidden_weights, mut hidden_biases, mut output_weights, mut output_bias) =
        init_params(window_size, hidden_size);

    println!("Training...");
    train(
        window_size.clone(),
        &mut hidden_weights,
        &mut hidden_biases,
        &mut output_weights,
        &mut output_bias,
    );

    println!("Evaluating...");
    evaluate(
        window_size,
        &mut hidden_weights,
        &mut hidden_biases,
        &mut output_weights,
        &mut output_bias,
    );
}
