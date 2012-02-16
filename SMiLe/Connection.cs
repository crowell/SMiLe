using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SMiLe
{
    [Serializable()]
    //a connection between two nodes
    public class Connection
    {
        private double weight; //weight of the connection
        private double deltaw; 
        private Node from; //node to the "left"
        private Node to; //node to the "right"
        public Connection(Node from, Node to, double weight)
        {
            this.weight = weight;
            this.from = from;
            this.to = to;
        }
        public Node getFromNode()
        {
            return this.from;
        }
        public Node getToNode()
        {
            return this.to;
        }
        public double getWeight()
        {
            return this.weight;
        }
        public void setDeltaw(double deltaw)
        {
            this.deltaw = deltaw;
        }
        public void setWeight(double weight)
        {
            this.weight = weight;
        }
    }
}
