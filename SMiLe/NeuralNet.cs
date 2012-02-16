using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SMiLe
{
    [Serializable()]
    public class NeuralNet
    {
        private List<List<Node>> layers;
        public NeuralNet(int[] layers)
        {
            //Construct a network of layers.Length layers with layers[ii] nodes in each layers
            if (layers.Length < 2)
            {
                //die
                return;
            }
            this.layers = new List<List<Node>>(layers.Length);
            for (int ii = 0; ii < layers.Length; ii++)
            {
                List<Node> layer = new List<Node>(layers[ii]);
                for (int jj = 0; jj < layers[ii]; jj++)
                {
                    layer.Add(new Node(ii,jj,false));
                }
                this.layers.Add(layer);
            }
        }
        private void addConnection(int fromLayer, int fromPos, int toLayer, int toPos, double weight)
        {
            //Connect node to node
            List<Node> layerFrom = this.layers[fromLayer];
            List<Node> layerTo = this.layers[toLayer];
            addConnection(layerFrom, fromPos, layerTo, toPos, weight);
        }
        private void addConnection(List<Node> layerFrom, int fromPos, List<Node> layerTo, int toPos, double weight)
        {
            //connect node to node
            Node fromNode = layerFrom[fromPos];
            Node toNode = layerTo[toPos];
            addConnection(fromNode, toNode, weight);
        }
        private void addConnection(Node fromNode, Node toNode, double weight)
        {
            //connect node to node
            Connection cn = new Connection(fromNode, toNode, weight);
            fromNode.addOutputConnection(cn);
            toNode.addInputConnection(cn);
        }

        private void addThreshold(int layer, int pos, double weight)
        {
            //add a threshold input to a node
            addThreshold((this.layers[layer])[pos], weight);
        }
        private void addThreshold(Node node, double weight)
        {
            //add a threshod input to a node
            Node thresh = new Node(node.getLayer(), node.getPos(), true);
            thresh.setOutput(-1);
            Connection cn = new Connection(thresh, node, weight);
            thresh.addOutputConnection(cn);
            node.addInputConnection(cn);
        }
        public void connectAll()
        {
            //connect all of the nodes from the previous layer to the current layer
            Random random = new Random();
            for (int ii = 1; ii < this.layers.Count; ii++)
            {
                List<Node> currentLayer = this.layers[ii];
                List<Node> prevLayer = this.layers[ii - 1];
                int pp = 0;
                foreach (Node pNode in prevLayer)
                {
                    int cc = 0;
                    foreach (Node cNode in currentLayer)
                    {
                        addConnection(prevLayer, pp++, currentLayer, cc++, random.NextDouble()); //assign a random weighting
                    }
                }
                foreach (Node node in currentLayer)
                {
                    addThreshold(node, random.NextDouble()); //add a random threshold
                }
            }
        }
        public void train(double[][] input, double[][] output, double r)
        {
            for (int ii = 0; ii < input.Length; ii++)
            {
                trainDS(input[ii], output[ii], r); //train 
            }
        }
        private void trainDS(double[] input, double[] output, double r)
        {
            //back propagate, then reset new values for weighting
            double target; //actual target output from training data
            double actual;
            setInput(input); //initialze the inputs
            getOutput(); //and propagate it through
            int outputLayer = this.layers.Count - 1;
            for (int ii = outputLayer; ii >= 0; ii--)
            {
                for (int jj = 0; jj < this.layers[ii].Count; jj++)
                {
                    if (ii == outputLayer) //the output layer connects to nobody
                    {
                        //calculate \Beta_{z}
                        target = output[jj];
                        actual = this.layers[ii][jj].getOutput();
                        this.layers[ii][jj].setBeta(target - actual);
                    }
                    else
                    {
                        List<Connection> outputCxns = this.layers[ii][jj].getAllOutputConnections();
                        double Bj = 0;
                        foreach (Connection cn in outputCxns)
                        {
                            //calculate \Beta_{j}
                            double wjk = cn.getWeight();
                            double ok = cn.getToNode().getOutput();
                            double Bk = cn.getToNode().getBeta();
                            Bj += (wjk * ok * (1 - ok) * Bk);
                        }
                        this.layers[ii][jj].setBeta(Bj);
                    }
                }
            }
            for (int ii = 0; ii < this.layers.Count; ii++)
            {
                for (int jj = 0; jj < this.layers[ii].Count; ii++)
                {
                    List<Connection> outputConnections = this.layers[ii][jj].getAllOutputConnections();
                    foreach (Connection cn in outputConnections)
                    {
                        //update weights based on error
                        double deltaw;
                        Node from = cn.getFromNode();
                        double oi = from.getOutput();
                        Node to = cn.getToNode();
                        double oj = to.getOutput();
                        double Bj = to.getBeta();
                        deltaw = getError(oi, oj, r, Bj);
                        cn.setWeight(cn.getWeight() + deltaw);
                    }
                }
            }

        }
        private double getError(double oi, double oj, double r, double Bj)
        {
            return r * oj * oi * (1 - oj) * Bj;
        }
        private void setInput(double[] input)
        {
            for (int ii = 0; ii < this.layers[0].Count; ii++)
            {
                this.layers[0][ii].setInput(input[ii]);
                this.layers[0][ii].setOutput(input[ii]);
            }
        }
        private double[] getOutput()
        {
            //propagate the output through
            int end = this.layers.Count - 1;
            int size = this.layers[end].Count;
            double[] outputs = new double[size];
            for (int ii = 1; ii < this.layers.Count; ii++)
            {
                for (int jj = 0; jj < this.layers[ii].Count; jj++)
                {
                    List<Connection> LConn = this.layers[ii][jj].getInputConnections();
                    double inputValue = 0;
                    foreach (Connection cn in LConn)
                    {
                        Node from = cn.getFromNode();
                        double fromVal = from.getOutput();
                        double weight = cn.getWeight();
                        inputValue += (fromVal * weight);
                    }
                    this.layers[ii][jj].setInput(inputValue);
                    double output = this.layers[ii][jj].f(inputValue);
                    this.layers[ii][jj].setOutput(output);
                }
            }
            for (int ii = 0; ii < this.layers[end].Count; ii++)
            {
                outputs[ii] = this.layers[end][ii].getOutput();
            }
            return outputs;
        }
        public double[] evaluate(double[] input)
        {
            //set input then get output
            setInput(input);
            double[] output = getOutput();
            return output;
        }
        public double error(double[][] input, double[][] output)
        {
            //get the sum of errors for the input output vectors
            double error = 0;
            for (int ii = 0; ii < input.Length; ii++)
            {
                double[] results = evaluate(input[ii]);
                for (int jj = 0; jj < results.Length; jj++)
                {
                    error += ((results[jj] - output[ii][jj]) * (results[jj] - output[ii][jj]));
                }
            }
            error /= input.Length;
            error = Math.Pow(error, 0.5);
            return error;
        }
    }
}
