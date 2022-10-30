import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.framework.optimizers.Adam
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import java.nio.file.Files
import java.nio.file.Paths

private const val labelPath = "src/main/resources/train-labels.idx1-ubyte"
private const val imagePath = "src/main/resources/train-images.idx3-ubyte"
private const val inputImage = "inputImage"
private const val inputLabel = "inputLabel"
private const val train = "train"
private const val loss = "loss"

fun main() {
    val g = build()
    val sess = Session(g)

    val mnist = setMnist()
    for (i in (1..10000)) {
        val batch = mnist[i]
        sess.runner()
            .feed(inputImage, TFloat32.vectorOf(*batch.image.map { it.toFloat() }.toFloatArray()))
            .feed(inputLabel, TFloat32.scalarOf(batch.label.toFloat()))
            .addTarget(train)
            .fetch("Hoge")
            .run()[0]
            .let { println((it as TFloat32).getFloat()) }
    }
}

fun build(): Graph {
    val g = Graph()
    val tf = Ops.create(g)
    val image = tf.withName(inputImage).placeholder(TFloat32::class.java)
    val x1 = tf.reshape(image, tf.constant(intArrayOf(-1, 28, 28, 1)))

    val k0 = tf.variable(tf.random.truncatedNormal(tf.constant(intArrayOf(4, 4, 1, 10)), TFloat32::class.java))
    val x2 = tf.nn.conv2d(x1, k0, listOf(1, 3, 3, 1), "VALID")
    val x3 = tf.nn.relu(x2)
    val x4 = tf.nn.maxPool(
        x3,
        tf.constant(intArrayOf(1, 3, 3, 1)),
        tf.constant(intArrayOf(1, 2, 2, 1)),
        "VALID"
    )
    val x5 = tf.reshape(x4, tf.constant(intArrayOf(-1, 160)))

    val w1 = tf.variable(tf.zeros(tf.constant(intArrayOf(160, 40)), TFloat32::class.java))
    val b1 = tf.variable(tf.constant(FloatArray(40) { 0.1f }))

    val x6 = tf.math.add(tf.linalg.matMul(x5, w1), b1)

    val x7 = tf.nn.relu(x6)

    val w2 = tf.variable(tf.zeros(tf.constant(intArrayOf(40, 10)), TFloat32::class.java))
    val b2 = tf.variable(tf.constant(FloatArray(10) { 0.1f }))

    val x8 = tf.math.add(tf.linalg.matMul(x7, w2), b2)

    val y = tf.nn.softmax(x8)

    val labels = tf.withName(inputLabel).placeholder(TFloat32::class.java)
    val loss = tf.withName(loss).math.mul(tf.constant(-1f), tf.reduceSum(tf.math.mul(labels, tf.math.log(y)), tf.constant(0)))
    val optimizer = Adam(g, 0.001f, 0.9f, 0.999f, 1e-8f).minimize(loss, train)

    val predictionMatch = tf.math.equal(tf.math.argMax(y, tf.constant(1)), tf.math.argMax(labels, tf.constant(1)))
    val accuracy = tf.math.mean(tf.dtypes.cast(predictionMatch, TFloat32::class.java), tf.constant(0))

    return g
}

fun setMnist(): List<StudyData> {
    val labels = Files.readAllBytes(Paths.get(labelPath)).map { it.toUByte() }.drop(8)
    val images = Files.readAllBytes(Paths.get(imagePath))
        .map { it.toUByte() }
        .drop(16)
        .windowed(size = 784, step = 784)
    return labels.zip(images) { label, image -> StudyData(label, image) }
}

data class StudyData(
    val label: UByte,
    val image: List<UByte>
) {
    override fun toString(): String {
        return image.windowed(28, 28)
            .joinToString("\n") { it.joinToString { b -> if (b > 150u) "■" else " " } }
    }
}
