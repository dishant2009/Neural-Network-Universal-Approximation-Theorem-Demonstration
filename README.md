# Neural Network Universal Approximation Theorem Demonstration

This interactive web application demonstrates how neural networks can approximate virtually any continuous function, showcasing the Universal Approximation Theorem through real-time visualization.

## Overview

This tool allows users to:
- Configure neural network architectures with multiple hidden layers
- Train networks on various target functions
- Visualize the training process and results in real-time
- Compare different network architectures
- Experiment with different activation functions and hyperparameters

## How to Run

### Option 1: Direct Browser Usage
1. Save the HTML file as `nn-universal-approximation.html`
2. Open the file in any modern web browser (Chrome, Firefox, Safari, Edge)
3. No installation or server required - it runs entirely in the browser

### Option 2: Local Server (Optional)
```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server

# Then navigate to http://localhost:8000/nn-universal-approximation.html
```

## Features

### Network Architecture Configuration
- **Multiple Hidden Layers**: Add up to 5 hidden layers
- **Neurons per Layer**: Configure 1-100 neurons per layer
- **Dynamic Architecture**: Add/remove layers on the fly
- **Real-time Display**: See the network structure as you build it

### Training Parameters
- **Learning Rate**: 0.001 to 0.1 (controls training speed)
- **Epochs**: 100 to 5000 (number of training iterations)  
- **Activation Functions**: Tanh, ReLU, Sigmoid, Leaky ReLU
- **Data Points**: 20 to 300 training samples
- **Noise Level**: 0 to 0.5 (adds randomness to training data)

### Target Functions
1. **Sine + Linear**: `sin(20x) + 3x`
2. **Polynomial**: `x³ - 2x² + x`
3. **Gaussian**: `exp(-10(x-0.5)²)`
4. **Step Function**: Discontinuous step
5. **Sawtooth Wave**: Periodic triangular wave
6. **Complex**: `sin(10x) × exp(-2x)`
7. **Absolute Sine**: `|sin(5x)| + 0.1x`
8. **Composite**: Sum of multiple sine waves

## How It Works

### Neural Network Implementation

The application implements a fully-connected feedforward neural network with:

1. **Forward Propagation**:
   ```javascript
   // For each layer:
   z = W × input + b
   activation = f(z)  // f is the activation function
   // Output layer uses linear activation
   ```

2. **Backpropagation**:
   - Computes gradients using the chain rule
   - Updates weights using gradient descent
   - Learning rate controls update magnitude

3. **Weight Initialization**:
   - Xavier initialization for Tanh/Sigmoid
   - He initialization for ReLU variants
   - Prevents vanishing/exploding gradients

### Training Process

1. **Data Generation**:
   - Samples points uniformly from [0, 1]
   - Evaluates target function at each point
   - Adds Gaussian noise based on noise level

2. **Training Loop**:
   ```
   For each epoch:
     Shuffle training data
     For each data point:
       Forward pass: compute prediction
       Compute loss: (target - prediction)²
       Backward pass: compute gradients
       Update weights: W = W + η × gradient
     Record average loss
   ```

3. **Visualization Updates**:
   - Loss chart updates every 50 epochs
   - Prediction curve updates every 100 epochs
   - Final results displayed after training

### Architecture Comparison

The comparison feature trains multiple architectures on the same data:
- Shallow: 1 hidden layer, 10 neurons
- Medium: 2 hidden layers, 20 neurons each
- Deep: 4 hidden layers, 10 neurons each
- Wide: 1 hidden layer, 50 neurons
- Very Deep: 5 hidden layers, 5 neurons each

Results show:
- Total parameters per architecture
- Final training loss
- Visual comparison of approximations

## Technical Details

### Libraries Used
- **Chart.js 3.9.1**: For real-time plotting
- **Vanilla JavaScript**: No framework dependencies
- **HTML5 Canvas**: For chart rendering

### Browser Compatibility
- Chrome 60+
- Firefox 60+
- Safari 12+
- Edge 79+

### Performance Considerations
- Training runs in the main thread
- Large networks (>100 neurons) may cause UI lag
- Recommended: <50 total neurons for smooth interaction

## Understanding the Results

### Loss Chart (Right Panel)
- **Y-axis**: Mean Squared Error (log scale)
- **X-axis**: Training epochs
- **Interpretation**: Lower is better, should decrease over time

### Function Approximation (Left Panel)
- **Blue points**: Training data with noise
- **Red line**: Neural network prediction
- **Green dashed**: True function (no noise)

### Key Observations

1. **Underfitting**: Too few neurons/layers - network cannot capture complexity
2. **Overfitting**: Network memorizes noise instead of underlying pattern
3. **Convergence**: Loss plateaus when network reaches capacity
4. **Architecture Impact**: 
   - Wider networks: More parameters, faster learning
   - Deeper networks: Better feature hierarchies, harder to train

## Mathematical Background

The Universal Approximation Theorem states that a feedforward network with:
- At least one hidden layer
- Finite number of neurons
- Non-linear activation function

Can approximate any continuous function on a compact subset of R^n to arbitrary accuracy.

This tool demonstrates this theorem by showing how networks of various architectures can learn to approximate different target functions.

## File Structure

```
nn-universal-approximation.html
├── HTML Structure
│   ├── Configuration controls
│   ├── Architecture builder
│   └── Visualization canvases
├── CSS Styling
│   ├── Layout grid system
│   ├── Modal dialog styles
│   └── Responsive design
└── JavaScript
    ├── DeepNeuralNetwork class
    ├── Training algorithms
    ├── Visualization functions
    └── UI event handlers
```

## Contributing

To extend this tool:

1. **Add new target functions**: Update `targetFunctions` object
2. **Add activation functions**: Modify `activate()` and `activateDerivative()`
3. **Change architecture limits**: Update validation in `addLayer()`
4. **Improve training**: Implement momentum, Adam optimizer, etc.

## License

This educational tool is provided as-is for learning purposes. Feel free to use, modify, and distribute.

## References

1. Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function"
2. Hornik, K. (1991). "Approximation capabilities of multilayer feedforward networks"
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 6
