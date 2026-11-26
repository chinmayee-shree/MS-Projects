# Agatha Christie - Poirot Inspired Text Generation Using LSTM

## Project Overview

This project explores text generation using a Long Short-Term Memory (LSTM) neural network.

The model is trained on the text of Agatha Christie's *Poirot Investigates*, obtained from Project Gutenberg. It learns word sequences from the book and uses the learned patterns to generate new text based on a user-provided prompt.

The objective is to experiment with language modelling and understand how recurrent neural networks can learn sequential patterns in natural language.

## Project Objectives

- Prepare and clean a literary text dataset for language modelling.
- Build a word-level tokenizer and convert text into numerical sequences.
- Develop an LSTM-based neural network for next-word prediction.
- Train the model using a sequence-based learning approach.
- Generate text using different sampling techniques.
- Experiment with temperature, Top-K and Top-P sampling.

## Dataset

The project uses the text file `61262-0.txt`, obtained from Project Gutenberg.

The text is processed to remove the surrounding Gutenberg material and retain the relevant literary content.

### Text Preprocessing

The preprocessing pipeline includes:

- Converting text to lowercase.
- Normalizing quotation marks and punctuation.
- Removing unwanted characters.
- Cleaning extra whitespace.
- Tokenizing the text into words.
- Limiting the vocabulary to 10,000 tokens.

## Model Architecture

A sequential LSTM-based neural network is used for next-word prediction.

| Layer | Configuration |
|---|---|
| Input | 40 tokens |
| Embedding | 200 dimensions |
| LSTM 1 | 256 units, return sequences |
| Dropout 1 | 0.2 |
| LSTM 2 | 256 units |
| Dropout 2 | 0.2 |
| Output | Dense layer with 10,000 units and Softmax |

The model contains approximately 5.56 million trainable parameters.

### Training Configuration

| Parameter | Value |
|---|---|
| Sequence length | 40 |
| Maximum vocabulary | 10,000 |
| Batch size | 128 |
| Maximum epochs | 30 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Sparse Categorical Crossentropy |
| Validation split | 10% |
| Random seed | 42 |

Early stopping and model checkpointing are used to monitor validation loss and retain the best-performing model.

## Text Generation

After training, the best saved model and tokenizer are reloaded for text generation.

The generation function accepts a user-provided prompt and predicts subsequent words one at a time.

The following sampling techniques are implemented:

- **Temperature:** Controls the randomness of word selection.
- **Top-K sampling:** Restricts selection to the K most probable next tokens.
- **Top-P sampling:** Selects from a probability-based subset of likely tokens.

The generation process can stop when a sentence-ending punctuation mark is produced after a minimum number of generated words.

### Example Prompt

```text
hercule poirot said
```

The model generates a continuation based on the patterns learned during training.

## Training Results

The model generated 66,527 input-target sequences from the processed text.

Training stopped after 9 epochs through early stopping.

| Metric | Result |
|---|---:|
| Best validation loss | 5.8907 |
| Best validation epoch | 5 |
| Final training epoch | 9 |
| Trainable parameters | 5,563,280 |

The best validation loss was achieved during epoch 5. Training stopped after subsequent epochs failed to improve the validation loss.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pickle
- Google Colab
- Google Drive

## Repository Structure

```text
poirot-text-generation/
│
├── notebooks/
│   └── poirot_text_generation.ipynb
│
├── results/
│   ├── figures/
│   └── samples/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

1. Clone or download this repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook in Google Colab.
4. Upload the Project Gutenberg text file `61262-0.txt` to the expected Google Drive directory.
5. Update the dataset and output paths if necessary.
6. Run the notebook cells in order to preprocess the text and train the model.
7. Use the interactive prompt-based demonstration to generate text.

The trained model and tokenizer are saved as `poirot_model.keras` and `tokenizer.pkl`, respectively.

## Limitations and Future Improvements

- The model is trained on a single literary text, limiting the diversity of its learned vocabulary and language patterns.
- The generated text can contain grammatical inconsistencies and incoherent word sequences.
- The current implementation does not include an independent test-set evaluation.
- The model's performance could be explored using additional evaluation methods and larger text datasets.

Possible future improvements include experimenting with different sequence lengths, model architectures, training configurations and sampling strategies.

## Conclusion

This project demonstrates the use of an LSTM-based language model for next-word prediction and text generation.

By learning sequential word patterns from a literary text, the model can generate continuations from user-provided prompts. The project provides practical experience with text preprocessing, tokenization, recurrent neural networks and probabilistic text-generation techniques.
