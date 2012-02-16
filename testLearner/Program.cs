﻿using System;
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
            int[] layers = { 2 , 2,  2 };
            SMiLe.NeuralNet net = new NeuralNet(layers);
            double[][] inputvs2 = new double[4][];
            inputvs2[0] =new double[]  { 0, 0 };
            inputvs2[1] = new double[]{ 0, 1 };
            inputvs2[2] = new double[]{ 1, 1 };
            inputvs2[3] = new double[]{ 1, 0 };
            double[][] outputvs2 = new double[4][] ;
            outputvs2[0] =new double[]{ 0, 0 };
            outputvs2[1] =new double[] { 0, 1 };
            outputvs2[2] =new double[] { 1, 1 };
            outputvs2[3] = new double[]{ 0, 1 } ;

            double[][] testing = new double[2][];
            testing[0] = new double[] {0,0};
            testing[1] =new double[] {0,1};
            double[][] testout = new double[2][];
            testout[0] = new double[] {0,0};
            testout[1] = new double[] {0,1};

            for(int ii = 0; ii<100; ii++)
            {
                net.train(inputvs2, outputvs2, 3);
                System.Console.WriteLine(net.error(inputvs2,outputvs2));
            }
            double[] testme = {0,1};
            
            double[] output = net.evaluate(testme);
            testme = new double[]{0,0};
            output = net.evaluate(testme);
            testme =new double[] {0,1};
            output = net.evaluate(testme);
            testme = new double[]{1,1};
            output = net.evaluate(testme);
            testme = new double[] { 1, 0 };
            output = net.evaluate(testme);
            double rate = net.errorrate(inputvs2, outputvs2);

            net.SAVE("neural.net", net);

            NeuralNet net2 = new NeuralNet(layers);
            net2 = net2.LOAD("neural.net");
            NeuralNet net34 = new NeuralNet().LOAD("neural.net");
            output = net2.evaluate(testme);

            //System.Console.WriteLine(net.error(testing, testout));

        }
    }
}
