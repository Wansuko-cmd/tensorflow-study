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

// SAME: -> 0埋めして画面の端までやる
// VALID: -> 画面の端はやらない
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

    // 画像の入力
    val input = tf
        // feedで指定してやることでここに画像の値が入る
        .withName(INPUT_NAME)
        .placeholder(
            TUint8::class.java,
            // [-1(要素数不明), 28, 28]の3次元データを用意
            // 1: 何枚目の画像か, 2: 画像の縦, 3: 画像の横
            Placeholder
                .shape(Shape.of(-1, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
        )
    // チャンネルの次元を追加（どんな色が出されるかを表す。今回は白黒のみなので1となる）
    val inputReshaped = tf.reshape(
        input,
        tf.array(
            -1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS,
        ),
    )
    // 画像の正解ラベルを取得(符号なし8ビット整数)
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

    /**
     * 畳み込みの重み（というかカーネル）設定
     * ランダムな重みに0.1をかけた値を用いる
     * variableなので学習時に変わる
     */
    val conv1Weights = tf.variable(
        tf.math.mul(
            /**
             * 切断正規分布に従って重みを初期化する(切断正規分布: 正規分布の端っこを切り落とした分布)
             */
            tf.random.truncatedNormal(
                // 次元の大きさ [高さ, 幅, 入力チャンネル数, 出力チャンネル数]
                // つまり今回は32個のランダムなカーネルが生成される
                tf.array(5, 5, NUM_CHANNELS, 32),
                TFloat32::class.java,
                // 乱数生成に用いられる。これにより再現性を出す
                TruncatedNormal.seed(SEED),
            ),
            tf.constant(0.1f),
        )
    )

    /**
     * 二次元畳み込み
     * 第一引数: 畳み込み対象
     * 第二引数: 畳み込みの際に用いるカーネル
     * 第三引数: 各次元ごとのずらす幅(stride) 今回は4次元なので4次元配列で渡す(画像数とチャンネル数は1で固定するとのこと)
     * 第４引数: 畳み込みの際に利用するアルゴリズム
     */
    val conv1 = tf.nn.conv2d(scaledInput, conv1Weights, listOf(1L, 1L, 1L, 1L), PADDING_TYPE)
    // バイアス値
    // 全て0の一次元配列(長さ32)とする
    // variableなので学習時に変わる
    val conv1Biases = tf.variable(
        tf.fill(
            tf.array(32),
            tf.constant(0.0f),
        ),
    )
    // バイアス値を足した後の32次元の出力に対してreluを適用
    val relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases))

    // プーリングを実行
    // 最大値をとっていく
    // 出力値自体は小さくなるが、出力された数は同じ32個である
    val pool1 = tf.nn.maxPool(
        relu1,
        // 2 × 2 のマス目から取っていく
        tf.array(1, 2, 2, 1),
        // 2 × 2個飛ばし
        tf.array(1, 2, 2, 1),
        // 端っこまでやる
        PADDING_TYPE,
    )

    // ２度目の畳み込み用のカーネル
    val conv2Weight = tf.variable(
        tf.math.mul(
            // 先ほどの畳み込みの出力が32次元だったため、入力は32個
            // 出力は64個となる
            tf.random.truncatedNormal(
                tf.array(5, 5, 32, 64),
                TFloat32::class.java,
                TruncatedNormal.seed(SEED),
            ),
            tf.constant(0.1f),
        ),
    )
    // ２度目の畳み込み
    val conv2 = tf.nn.conv2d(pool1, conv2Weight, listOf(1L, 1L, 1L, 1L), PADDING_TYPE)
    // バイアス値（出力が64個あるから64個用意）
    val conv2Biases = tf.variable(
        tf.fill(
            tf.array(64),
            tf.constant(0.1f),
        ),
    )
    // バイアスを適用後、reluを適用
    val relu2 = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases))

    // 最大値を取るプーリングを実行
    val pool2 = tf.nn.maxPool(
        relu2,
        tf.array(1, 2, 2, 1),
        tf.array(1, 2, 2, 1),
        PADDING_TYPE,
    )

    // 次元を書き換える
    // [一次元目のサイズ, 二次元目のサイズ, -1]となることが想定される
    // つまり最後の次元に全部詰め込む
    val flatten = tf.reshape(
        pool2,
        tf.concat(
            //
            listOf(
                // 0 ~ 1番目のサイズのみ取り出す
                tf.slice(tf.shape(pool2), tf.array(0), tf.array(1)),
                tf.array(-1),
            ),
            // 最初の次元で合体させる
            tf.constant(0),
        ),
    )

    // カーネル生成
    val fc1Weights = tf.variable(
        tf.math.mul(
            tf.random.truncatedNormal(
                // 2次元(画像のピクセル数 × 4)(512)
                tf.array(IMAGE_SIZE * IMAGE_SIZE * 4, 512),
                TFloat32::class.java,
                TruncatedNormal.seed(SEED),
            ),
            tf.constant(0.1f),
        )
    )
    // バイアス
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
