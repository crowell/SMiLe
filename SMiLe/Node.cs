using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters;
using System.Runtime.Serialization.Formatters.Binary;

namespace SMiLe
{
    [Serializable()]
    public class Node
    {
        //connections to and from the node
        private List<Connection> inputConnections;
        private List<Connection> outputConnections;
        //input to the node, output from the node
        private double input;
        private double output;
        //the "beta" of the particular node
        private double beta;
        //position in the net
        private int pos;
        private int layer;
        public Node(int layer, int pos, bool isThresh)
        {
            //initialize the node with its given layer and position
            this.pos = pos;
            this.layer = layer;
            this.inputConnections = new List<Connection>();
            this.outputConnections = new List<Connection>();
            this.beta = 0;
            this.input = 0;
            this.output = 0;
        }
        public void addInputConnection(Connection cn)
        {
            this.inputConnections.Add(cn);
        }
        public void addOutputConnection(Connection cn)
        {
            this.outputConnections.Add(cn);
        }
        public double getOutput()
        {
            return this.output;
        }
        public void setOutput(double oput)
        {
            this.output = oput;
        }
        public double getBeta()
        {
            return this.beta;
        }
        public void setBeta(double beta)
        {
            this.beta = beta;
        }
        public double f(double sigma)
        {
            return (1 / (1 + Math.Exp(-1 * sigma)));
        }
        public int getPos()
        {
            return this.pos;
        }
        public int getLayer()
        {
            return this.layer;
        }
        public void setInput(double input)
        {
            this.input = input;
        }
        public Connection getOutputConnection(int index)
        {
            return this.outputConnections[index];
        }
        public List<Connection> getAllOutputConnections()
        {
            return this.outputConnections;
        }
        public List<Connection> getInputConnections()
        {
            return this.inputConnections;
        }
    }
}
