using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SMiLe;

namespace testLearner
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] layers = { 2, 2, 6, 2 };
            SMiLe.NeuralNet net = new NeuralNet(layers);
            List<List<double>> inputvs2 = new List<List<double>>();
            inputvs2.Add(new List<double>() { 0, 0 });
            inputvs2.Add(new List<double>(){ 0, 1 });
            inputvs2.Add(new List<double>(){ 1, 1 });
           // inputvs2.Add(new List<double>() { 1, 0 });
            
            List<List<double>> outputvs2 = new List<List<double>>();
            outputvs2.Add(new List<double>() { 0, 0 });
            outputvs2.Add(new List<double>(){ 0, 1 });
            outputvs2.Add(new List<double>(){ 1, 1 });
           // outputvs2.Add(new List<double>() { 1, 0 });
            

            List<List<double>> testing = new List<List<double>>();
            List<List<double>> testout = new List<List<double>>();
            testing.Add(new List<double>() { 1, 0 });
            testout.Add(new List<double>() { 1, 0 });


            for (int ii = 0; ii < 10; ii++)
            {
                net.train(inputvs2, outputvs2, 3);
                System.Console.WriteLine(net.error(inputvs2, outputvs2));
            }

            List<double> output = net.evaluate(testing[0]);
 
 
            net.SAVE("neural.net", net);

            NeuralNet net2 = new NeuralNet(layers);
            net2 = net2.LOAD("neural.net");
            NeuralNet net34 = new NeuralNet().LOAD("neural.net");

            //System.Console.WriteLine(net.error(testing, testout));

        }
    }
}
