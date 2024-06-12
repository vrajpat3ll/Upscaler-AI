# Design Choices

1. Used malloc instead of calloc, previously had done otherwise.

    1. Because i realised that there was an issue with the * operator that would give garbage values.
    2. Fixed it by adding one extra line of `ans[i][j] = 0;`

1. Did not make a default constructor for NeuralNetwork.
    - There is not much reason for this, I just felt that `NueralNetwork.init(architecture)` would make more sense as we were initializing the NeuralNetwork.