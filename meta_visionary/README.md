# MetaVisionary

## What the Algorithm Does?

MetaVisionary is a Python package designed to implement meta-learning algorithms tailored for few-shot learning scenarios. It empowers users to train machine learning models efficiently even when confronted with datasets containing a limited number of samples and classes that bear resemblance to each other.

## Parameters

- `train_data_dir`: The directory containing the training data.
- `test_data_dir`: The directory containing the test data.
- `num_classes`: The number of classes in the dataset.
- `learning_rate`: The learning rate for the optimizer.
- `meta_step_size`: The meta-learning step size.
- `inner_batch_size`: Batch size for inner loop training.
- `eval_batch_size`: Batch size for evaluation.
- `meta_iters`: Number of meta-iterations.
- `eval_iters`: Number of evaluation iterations.
- `inner_iters`: Number of inner loop iterations.
- `eval_interval`: Interval for evaluation.
- `train_shots`: Number of shots for training.
- `shots`: Number of shots for evaluation.
- `img_height`: Height of input images.
- `img_width`: Width of input images.

## Installation

You can install MetaVisionary using the following command:

```python
pip install https://github.com/subhayudutta/FewShotMetaLearning/releases/download/v0.1.0/meta_visionary-0.1-py3-none-any.whl
```

## Importing

To use MetaVisionary, you can import the `meta_learning` function:

```python
from meta_visionary.few_shot_meta import meta_learning
```

## Calling the Function
To call the meta_learning function, simply provide the directories containing the training and test data, along with the number of classes, like so:

```python
def main():
    train_data_dir = 'PATH'
    test_data_dir = 'PATH'
    meta_learning(train_data_dir, test_data_dir, num_classes=2, learning_rate=0.003, meta_step_size=0.25,
                  inner_batch_size=25, eval_batch_size=25, meta_iters=2000, eval_iters=5, inner_iters=4,
                  eval_interval=1, train_shots=20, shots=5, img_height=28, img_width=28)

if __name__ == "__main__":
    main()

```
The default values for the parameters of the meta_learning function are provided. You can adjust these values according to your specific requirements when calling the function.

## Citation
If you use MetaVisionary in your research work, please consider citing the following paper:

#### "If Human Can Learn from Few Samples, Why Can’t AI? An Attempt On Similar Object Recognition With Few Training Data Using Meta-Learning" [Link to Paper](https://ieeexplore.ieee.org/document/10396424)

```
@inproceedings{dutta2023if,
  title={If Human Can Learn from Few Samples, Why Can’t AI? An Attempt On Similar Object Recognition With Few Training Data Using Meta-Learning},
  author={Dutta, Subhayu and Goswami, Saptiva and Debnath, Sonali and Adhikary, Subhrangshu and Majumder, Anandaprova},
  booktitle={2023 IEEE North Karnataka Subsection Flagship International Conference (NKCon)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```


## Support and Code of Conduct
If you encounter any issues with MetaVisionary or have any questions, feel free to reach out to us via GitHub issues. We welcome contributions and feedback from the community.