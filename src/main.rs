use std::{error::Error, fs::File};

#[derive(Debug, Clone)]
struct Row {
    ticker: String,
    _index: usize,
    price: f64,
    is_minima: bool,
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
            is_minima: record[3].parse::<u8>()? == 1,
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

fn forward_pass() {
    /*
     * hidden_weights size: hidden_layer_neurons_count * input_dimensions,
     * 50 * 100,
     * 50 rows, 100 columns,
     * there is 50 neuron and each neuron has 100 inputs,
     */
    let hidden_weights: Vec<Vec<f64>> = Vec::new();
    let input_vector: Vec<f64> = Vec::new();
    let hidden_biases: Vec<f64> = Vec::new();

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
}

fn sigmoid() {}

fn loss() {}

fn backward_pass() {}

fn train() {}

fn predict() {}

fn evaluate() {}

fn main() {
    let mut records: Vec<Row> = get_data().unwrap();
    pre_process_data(&mut records);

    forward_pass();
}
