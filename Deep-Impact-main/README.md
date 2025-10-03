## Introduction

**DeepImpact** is a comprehensive Python tool developed as part of a group project to simulate the atmospheric entry of asteroids and predict their potential airburst damage, specifically focusing on impacts over the UK. This project was inspired by historical events like the Chelyabinsk meteor and employs numerical methods to solve differential equations that model these celestial encounters. Our aim is to provide an accessible, accurate, and educational simulator for both academic and enthusiast communities.

## Group Contributions

This project was a collaborative effort, each member contributing to various facets including algorithm development, data analysis, testing, and documentation. We encourage you to view individual contributions in the commit history.

## Installation

To install the module and any pre-requisites, from the base directory run
```
pip install -r requirements.txt
pip install -e .
```  

## Data Setup
To download the necessary postcode data for mapping impact scenarios:

```
python download_data.py
```

## Automated testing

To run the pytest test suite, from the base directory run
```
pytest tests/
```

Note that you should keep the tests provided, adding new ones as you develop your code. If any of these tests fail it is likely that the scoring algorithm will not work.

## Documentation

To generate the documentation (in html format)
```
python -m sphinx docs html
```

See the `docs` directory for the preliminary documentation provided that you should add to.

## Example usage

For example usage see `example.py` in the examples folder:
```
python examples/example.py
```

## Project Notebooks

For a deep dive into the project specifications, methodologies, and examples, check out the Python notebooks:

- `ProjectDescription.ipynb`
- `AirburstSolver.ipynb`
- `DamageMapper.ipynb`

These notebooks provide the context, mathematical background, and interactive examples for understanding and using the asteroid impact simulation tool.

## Feedback and Contributions

We welcome feedback and contributions to the DeepImpact project. If you have suggestions, bug reports, or would like to contribute, please refer to the issues section or submit a pull request.