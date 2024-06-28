# Design Choices

1. Used malloc instead of calloc, previously had done otherwise.

    - Because i realised that there was an issue with the * operator that would give garbage values.
    - Fixed it by adding one extra line of `ans[i][j] = 0;`

1. Did not make a default constructor for NeuralNetwork.
    - There is not much reason for this, I just felt that `NueralNetwork.init(architecture, activationFunctions)` would make more sense as we were initializing the NeuralNetwork.

1. Have put up different activation functions for each layer.
    - Could have made it for each neuron, but felt tedious.
    