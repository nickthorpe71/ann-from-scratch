const mmap = math.map; // use to pass each element of a metric to a function
const rand = math.rand;
const transp = math.transpose;
const mat = math.matrix;
const e = math.evaluate;
const sub = math.subtract;
const sqr = math.square;
const sum = math.sum;

// The neural network class
class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate, wih, who) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;

        this.wih = wih || sub(mat(rand([hiddenNodes, inputNodes])), 0.5);
        this.who = who || sub(mat(rand([outputNodes, hiddenNodes])), 0.5);

        this.activationFunction = function (x) {
            return 1 / (1 + Math.exp(-x));
        };
    }

    cache = { loss: [] };

    static normalizeData = (data) => {
        /*...*/
    };

    forward = (input) => {
        /*...*/
    };
    backward = (input, target) => {
        /*...*/
    };
    update = () => {
        /*...*/
    };
    predict = (input) => {
        /*...*/
    };
    train = (input, target) => {
        /*...*/
    };
}
