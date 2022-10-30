import dataset.MnistDataset
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.framework.optimizers.AdaDelta
import org.tensorflow.framework.optimizers.AdaGrad
import org.tensorflow.framework.optimizers.AdaGradDA
import org.tensorflow.framework.optimizers.Adam
import org.tensorflow.framework.optimizers.GradientDescent
import org.tensorflow.framework.optimizers.Momentum
import org.tensorflow.framework.optimizers.RMSProp
import org.tensorflow.ndarray.FloatNdArray
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.index.Indices
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TUint8
import java.util.Locale

private const val PIXEL_DEPTH = 255
private const val NUM_CHANNELS = 1
private const val IMAGE_SIZE = 28
private const val NUM_LABELS = MnistDataset.NUM_CLASSES
private const val SEED = 123456789L

private const val PADDING_TYPE = "SAME"

private const val INPUT_NAME = "input"
private const val OUTPUT_NAME = "output"
private const val TARGET = "target"
private const val TRAIN = "train"
private const val TRAINING_LOSS = "training_loss"

private const val TRAINING_IMAGES_ARCHIVE = "train-images-idx3-ubyte.gz"
private const val TRAINING_LABELS_ARCHIVE = "train-labels-idx1-ubyte.gz"
private const val TEST_IMAGES_ARCHIVE = "t10k-images-idx3-ubyte.gz"
private const val TEST_LABELS_ARCHIVE = "t10k-labels-idx1-ubyte.gz"

fun main() {
    println("Usage: MNISTTest <num-epochs> <minibatch-size> <optimizer-name>")
    val dataset = MnistDataset.create(
        0,
        TRAINING_IMAGES_ARCHIVE,
        TRAINING_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
    )

    println("Loaded data.")

    val epochs = 2
    val minibatchSize = 50
    val graph = build("adam")
    val session = Session(graph)
    train(session, epochs, minibatchSize, dataset)
    println("Trained model")
    test(session, minibatchSize, dataset)
}

fun build(optimizerName: String): Graph {
    val graph = Graph()

    val tf = Ops.create(graph)
    val input = tf
        .withName(INPUT_NAME)
        .placeholder(
            TUint8::class.java,
            Placeholder
                .shape(Shape.of(-1, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
        )
    val inputReshaped = tf.reshape(input, tf.array(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    val labels = tf.withName(TARGET).placeholder(TUint8::class.java)
    val centeringFactor = tf.constant(PIXEL_DEPTH / 2.0f)
    val scalingFactor = tf.constant(PIXEL_DEPTH.toFloat())
    val scaledInput = tf.math.div(
        tf.math.sub(
            tf.dtypes.cast(inputReshaped, TFloat32::class.java),
            centeringFactor,
        ),
        scalingFactor,
    )

    val conv1Weights = tf.variable(
        tf.math.mul(
            tf.random.truncatedNormal(
                tf.array(5, 5, NUM_CHANNELS, 32),
                TFloat32::class.java,
                TruncatedNormal.seed(SEED),
            ),
            tf.constant(0.1f),
        )
    )
    val conv1 = tf.nn.conv2d(scaledInput, conv1Weights, listOf(1L, 1L, 1L, 1L), PADDING_TYPE)
    val conv1Biases = tf.variable(
        tf.fill(
            tf.array(32),
            tf.constant(0.0f),
        ),
    )
    val relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases))

    val pool1 = tf.nn.maxPool(
        relu1,
        tf.array(1, 2, 2, 1),
        tf.array(1, 2, 2, 1),
        PADDING_TYPE,
    )

    val conv2Weight = tf.variable(
        tf.math.mul(
            tf.random.truncatedNormal(
                tf.array(5, 5, 32, 64),
                TFloat32::class.java,
                TruncatedNormal.seed(SEED),
            ),
            tf.constant(0.1f),
        ),
    )
    val conv2 = tf.nn.conv2d(pool1, conv2Weight, listOf(1L, 1L, 1L, 1L), PADDING_TYPE)
    val conv2Biases = tf.variable(
        tf.fill(
            tf.array(64),
            tf.constant(0.1f),
        ),
    )
    val relu2 = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases))

    val pool2 = tf.nn.maxPool(
        relu2,
        tf.array(1, 2, 2, 1),
        tf.array(1, 2, 2, 1),
        PADDING_TYPE,
    )

    val flatten = tf.reshape(
        pool2,
        tf.concat(
            listOf(tf.slice(tf.shape(pool2), tf.array(0), tf.array(1)), tf.array(-1)),
            tf.constant(0),
        ),
    )

    val fc1Weights = tf.variable(
        tf.math.mul(
            tf.random.truncatedNormal(
                tf.array(IMAGE_SIZE * IMAGE_SIZE * 4, 512),
                TFloat32::class.java,
                TruncatedNormal.seed(SEED),
            ),
            tf.constant(0.1f),
        )
    )
    val fc1Biases = tf.variable(tf.fill(tf.array(512), tf.constant(0.1f)))
    val relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases))

    val fc2Weights = tf.variable(
        tf.math.mul(
            tf.random.truncatedNormal(
                tf.array(512, NUM_LABELS),
                TFloat32::class.java,
                TruncatedNormal.seed(SEED),
            ),
            tf.constant(0.1f),
        )
    )

    val fc2Biases = tf.variable(tf.fill(tf.array(NUM_LABELS), tf.constant(0.1f)))

    val logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases)

    val prediction = tf.withName(OUTPUT_NAME).nn.softmax(logits)

    val oneHot = tf.oneHot(labels, tf.constant(10), tf.constant(1.0f), tf.constant(0.0f))
    val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, oneHot)
    val labelLoss = tf.math.mean(batchLoss.loss(), tf.constant(0))
    val regularizers = tf.math.add(
        tf.nn.l2Loss(fc1Weights),
        tf.math.add(
            tf.nn.l2Loss(fc1Biases),
            tf.math.add(
                tf.nn.l2Loss(fc2Weights),
                tf.nn.l2Loss(fc2Biases),
            ),
        ),
    )
    val loss = tf.withName(TRAINING_LOSS).math.add(labelLoss, tf.math.mul(regularizers, tf.constant(5e-4f)))

    val optimizer = when (optimizerName.lowercase(Locale.getDefault())) {
        "adadelta" -> AdaDelta(graph, 1f, 0.95f, 1e-8f)
        "adagradda" -> AdaGradDA(graph, 0.01f)
        "adagrad" -> AdaGrad(graph, 0.01f)
        "adam" -> Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f)
        "sgd" -> GradientDescent(graph, 0.01f)
        "momentum" -> Momentum(graph, 0.01f, 0.9f, false)
        "rmsprop" -> RMSProp(graph, 0.01f, 0.9f, 0.0f, 1e-10f, false)
        else -> throw Exception()
    }
    println("Optimizer = $optimizer")
    val minimize = optimizer.minimize(loss, TRAIN)

    return graph
}

fun train(
    session: Session,
    epochs: Int,
    minibatchSize: Int,
    dataset: MnistDataset,
) {
    var interval = 0
    for (i in 0 until epochs) {
        for (trainingBatch in dataset.trainingBatches(minibatchSize)) {
            val batchImages = TUint8.tensorOf(trainingBatch.images)
            val batchLabels = TUint8.tensorOf(trainingBatch.labels)
            session.runner()
                .feed(TARGET, batchLabels)
                .feed(INPUT_NAME, batchImages)
                .addTarget(TRAIN)
                .fetch(TRAINING_LOSS)
                .run()[0]
                .let {
                    if (interval % 100 == 0) {
                        println("Iteration = $interval, training loss = ${(it as TFloat32).getFloat()}")
                    }
                }
            interval++
        }
    }
}

fun test(
    session: Session,
    minibatchSize: Int,
    dataset: MnistDataset,
) {
    var correctCount = 0
    val confusionMatrix = Array(10) { IntArray(10) }

    for (trainingBatch in dataset.testBatches(minibatchSize)) {
        val transformedInput = TUint8.tensorOf(trainingBatch.images)
        val outputTensor = session.runner()
            .feed(INPUT_NAME, transformedInput)
            .fetch(OUTPUT_NAME)
            .run()[0] as TFloat32

        val labelBatch = trainingBatch.labels
        for (k in 0 until labelBatch.shape().size(0)) {
            val trueLabel = labelBatch.getByte(k)
            val predLabel = argmax(outputTensor.slice(Indices.at(k), Indices.all()))
            if (predLabel == trueLabel.toInt()) {
                correctCount++
            }
            confusionMatrix[trueLabel.toInt()][predLabel]++
        }
    }

    println("Final accuracy = ${correctCount.toFloat() / dataset.numTestingExamples()}")

    val sb = with(StringBuilder()) {
        append("Label")
        (confusionMatrix.indices).forEach { append(String.format("%1\$5s", "" + it)) }
        append("\n")
        (confusionMatrix.indices).forEach { i ->
            append(String.format("%1$5s", "" + i))
            (confusionMatrix[i].indices).forEach { j ->
                append(String.format("%1$5s", "" + confusionMatrix[i][j]))
            }
        }
        append("\n")
    }
    println(sb)
}

fun argmax(probabilities: FloatNdArray): Int {
    var maxVal = Float.NEGATIVE_INFINITY
    var idx = 0
    for (i in 0 until probabilities.shape().size(0)) {
        val curVal = probabilities.getFloat(i)
        if (curVal > maxVal) {
            maxVal = curVal
            idx = i.toInt()
        }
    }
    return idx
}
