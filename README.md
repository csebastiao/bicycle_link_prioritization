# Bicycle_link_prioritization

Because cycling is an effective solution for making urban transport more sustainable, many cities worldwide are currently announcing extensive bicycle network development plans. These plans typically outline hundreds of kilometers of new links to be built over several years. However, the links in most of these plans are not ordered, and it is unclear how to prioritize them to reach a functional network as fast as possible. For achieving this clarity, here we develop and analyze an automatized method that optimally prioritizes all links in a given bicycle network plan. We start with the full plan and iteratively delete links that minimize loss of a network metric (connectedness, directness, coverage). This greedy, subtractive process ensures an optimal build-up of functional network structures, identifying the best time-ordered investment plan for the development budget.

## To add as a local package

Based on the [The Good Research Code Handbook](https://goodresearch.dev/setup.html#pip-install-your-package), you should:
* Put yourself on your virtual environment and on the folder, such as /bicycle_link_prioritization
* Run pip install -e .

". indicates that we’re installing the package in the current directory. -e indicates that the package should be editable. That means that if you change the files inside the [source] folder, you don’t need to re-install the package for your changes to be picked up by Python."
