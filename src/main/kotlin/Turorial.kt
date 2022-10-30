import org.tensorflow.op.Ops
import org.tensorflow.op.image.DecodePng
import org.tensorflow.types.TFloat32

fun tutorial() {
    val tf = Ops.create()
    val fileName = tf.constant("src/main/resources/img.png")
    val readFile = tf.io.readFile(fileName)
    val image = tf.image.decodePng(readFile, arrayOf(DecodePng.channels(1)))
    val floatImage = tf.dtypes.cast(image, TFloat32::class.java)

    val reshape = tf.reshape(floatImage, tf.constant(intArrayOf(-1, 32, 32, 1)))
    val kernel = tf.constant(
        arrayOf(
            floatArrayOf(0f, -1f, -1f, -1f, 0f),
            floatArrayOf(-1f, 0f, 3f, 0f, -1f),
            floatArrayOf(-1f, 3f, 0f, 3f, -1f),
            floatArrayOf(-1f, 0f, 3f, 0f, -1f),
            floatArrayOf(0f, -1f, -1f, -1f, 0f)
        )
    )
    val kernelReshape = tf.reshape(kernel, tf.constant(intArrayOf(5, 5, 1, 1)))
    val result = tf.nn.conv2d(reshape, kernelReshape, listOf(1, 3, 3, 1), "VALID")
    val r = tf.reshape(result, tf.constant(intArrayOf(10, 10)))
    println(r.asTensor().scalars().forEach { println(it.getFloat()) })

    val y = tf.constant(floatArrayOf(1.1f, 3.2f, -0.9f))
    val p = tf.nn.softmax(y)
    println(p.asTensor().scalars().forEach { println(it.getFloat()) })
}
