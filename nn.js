const mmap = math.map; // use to pass each element of a metric to a function
const rand = math.rand;
const transp = math.transpose;
const mat = math.matrix;
const e = math.evaluate;
const sub = math.subtract;
const sqr = math.square;
const sum = math.sum;

/*
The last two parameters wih and who stands for “weights of input-to-hidden layer” as well as “weights of hidden-to-output layer”. They are optional: if they are passed then the neural network assumes that you want to initiate it with already trained weights, if not, then it will initiate the weights randomly.
*/
class NeuralNetwork {
    constructor(inputnodes, hiddennodes, outputnodes, learningrate, wih, who) {
        this.inputnodes = inputnodes;
        this.hiddennodes = hiddennodes;
        this.outputnodes = outputnodes;
        this.learningrate = learningrate;

        this.wih = wih || sub(mat(rand([hiddennodes, inputnodes])), 0.5);
        this.who = who || sub(mat(rand([outputnodes, hiddennodes])), 0.5);

        this.act = (matrix) => mmap(matrix, (x) => 1 / (1 + Math.exp(-x)));
    }

    cache = { loss: [] };

    static normalizeData = (data) => {
        /*...*/
    };

    forward = (input) => {
        const wih = this.wih;
        const who = this.who;
        const act = this.act;

        input = transp(mat([input]));

        // forward prop from the input layer to the hidden layer by building
        // the dot product of our matrices weightsInputToHidden and input
        const h_in = e("wih * input", { wih, input });
        const h_out = act(h_in);

        // then do the forward prop from the hidden layer to the output layer
        // with the same principles applied
        const o_in = e("who * h_out", { who, h_out });
        const actual = act(o_in);

        // input, h_out and the actual prediction value actual are all stored in
        //  cache because we need them for the backward method
        this.cache.input = input;
        this.cache.h_out = h_out;
        this.cache.actual = actual;

        return actual;
    };
    backward = (input, target) => {
        const who = this.who;
        const input = this.cache.input;
        const h_out = this.cache.h_out;
        const actual = this.cache.actual;

        target = transp(mat([target]));

        // calculate the gradient of the error function (E) w.r.t the activation function (A)
        const dEdA = sub(target, actual);

        // calculate the gradient of the activation function (A) w.r.t the weighted sums (Z) of the output layer
        const o_dAdZ = e("actual .* (1 - actual)", {
            actual,
        });

        // calculate the error gradient of the loss function w.r.t the weights of the hidden-to-output layer
        const dwho = e("(dEdA .* o_dAdZ) * h_out'", {
            dEdA,
            o_dAdZ,
            h_out,
        });

        // calculate the weighted error for the hidden layer
        const h_err = e("who' * (dEdA .* o_dAdZ)", {
            who,
            dEdA,
            o_dAdZ,
        });

        // calculate the gradient of the activation function (A) w.r.t the weighted sums (Z) of the hidden layer
        const h_dAdZ = e("h_out .* (1 - h_out)", {
            h_out,
        });

        // calculate the error gradient of the loss function w.r.t the weights of the input-to-hidden layer
        const dwih = e("(h_err .* h_dAdZ) * input'", {
            h_err,
            h_dAdZ,
            input,
        });

        this.cache.dwih = dwih;
        this.cache.dwho = dwho;
        this.cache.loss.push(sum(sqr(dEdA)));
    };
    update = () => {
        const wih = this.wih;
        const who = this.who;
        const dwih = this.cache.dwih;
        const dwho = this.cache.dwho;
        const r = this.learningrate;

        this.wih = e("wih + (r .* dwih)", { wih, r, dwih });
        this.who = e("who + (r .* dwho)", { who, r, dwho });
    };
    predict = (input) => {
        /*...*/
    };
    train = (input, target) => {
        /*...*/
    };
}
