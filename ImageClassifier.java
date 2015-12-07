package org.deeplearning4j.examples.convolution;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ImageClassifier {

    public static void main(String[] args) throws IOException, InterruptedException {

        // Parameters for the Neural Network!
        int nChannels = 1;
        int outputNum = 2;
        int iterations = 5;
        int seed = 123;
        int batchSize = 500;
        int numRows = 28;
        int numColumns = 28;


        // Path to the labeled images. This depends on the system!
        String path =  "/home/bread/IW/Training_Data"; //I moved a lot of the training data to another folder!
        //Create arrays to hold the labels
        List<String> labels = new ArrayList<>();

        String testingPath = "/home/bread/IW/Testing_Data";
        labels.add("Healthy");
        labels.add("Infected");


        // Instantiating a RecordReader pointing to the data path with the specified
        // height and width for each image.
        //Creates a new RecordReader object. It points to the datapath.
        RecordReader healthyReader = new ImageRecordReader(28, 28, true, labels);
        healthyReader.initialize(new FileSplit(new File(path)));

        RecordReader testingReader = new ImageRecordReader(28, 28, true, labels);
        testingReader.initialize(new FileSplit(new File(path)));

        // Canova to Dl4j
        DataSetIterator healthyIterator = new RecordReaderDataSetIterator(healthyReader, 784, labels.size());
        DataSetIterator duplicate = new RecordReaderDataSetIterator(testingReader, 784, labels.size());
        //My new builder
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .stride(2,2)
                        .nIn(nChannels)
                        .nOut(2)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
                        .build())
               // .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                //        .nOut(outputNum)
                //        .activation("softmax")
                //        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

        MultiLayerConfiguration conf = builder.build();

        //Set up the network
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Train
        System.out.println("******************* BEGINNING TRAINING ********************");
        while(healthyIterator.hasNext()){
            DataSet next = healthyIterator.next();
            network.fit(next);
            //System.out.println("butts");
        }
        System.out.println("******************* DONE WITH TRAINING ********************");
        System.out.println("　　　　　　　　 　 　 　 ,, 、\n" +
                "　　　　　　　　　　　　/: : l　　,, -､\n" +
                "　　　　　　　 　 　 , : :' ':'└ノﾞ　:;;ﾘ\n" +
                "　　　　　　 　 　､'　 .: : : : : ミ .::;/　　　　　ﾍｯﾍｯﾍｯﾍｯ\n" +
                "　　 　 　 　 　 ノ゛ ゎ::.: :'¨ﾞ::.: :;八\n" +
                "　　 　 　 　 rｯ　　　 ,_　　　:: : : : ﾞ ,\n" +
                "　　 　 　 　 `ｩ--イヌﾞ　　　　:　　　' ,\n" +
                "　 　 　 　 　 (_／7´ 　 　 　 　　..:: : : ::､\n" +
                "　　　　　　　　　　ﾞ}　　　　　　,,.: : : : : : ゛゛ﾞﾞﾞ'' ､ _\n" +
                "　　　　　　　　　　 l,　 　 , : : : : :: : : : : : : : : : : : : : ‐- .\n" +
                "　　　　　　 　 　 　 l,　　' ': : : : : : : : : : : : : : : : : : : : : : : :ﾞ'.､\n" +
                "　　 　 　 　 　 　 　 ヾ　　: : : : : :.:;; : : : : : : : : : : : : : : : : : :.::.\n" +
                "　　　　　　　　　 　 　 丶　ﾞ ､: : : : :;;: : : : : : : : : : : : : : : : : ; ;}\n" +
                "　　　 　 　 　 　 　 　 　 _j　　l:　 　 ;;:: : : : : : : : : : : : : ; ; ; ;;ﾐ\n" +
                "　　　　　　　　　　 　 ',´, , ,,,ノ l:　　彡;;; ; ; ; ; : : : : :: ; ; ; ;;;ﾐ`\n" +
                "　 　 　 　 　 　 　 　 　 　 , -‐'ﾞ　 ｿﾞ ' ' ' ' ' ' ' ' ﾞ ﾞ ﾞ ﾞ ﾞ ﾞ´\n" +
                "　　　　　　　　　 　 　 　 　`' ' ' '´");
        // Testing -- We're not doing split test and train
        // Using the same training data as test.
        healthyIterator.reset();
        Evaluation eval = new Evaluation();

        //Test on the healthy dataset
        int i = 0;
        while(duplicate.hasNext()){
            System.out.println("Test number " + ++i);
            DataSet next = duplicate.next();
            INDArray predict2 = network.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict2);
        }


        System.out.println(eval.stats());
    }
}