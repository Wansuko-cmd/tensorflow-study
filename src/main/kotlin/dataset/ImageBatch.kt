package dataset

import org.tensorflow.ndarray.ByteNdArray
import org.tensorflow.ndarray.index.Indices.range
import java.lang.Long.min

data class ImageBatch(
    val images: ByteNdArray,
    val labels: ByteNdArray,
)

class ImageBatchIterator(
    val batchSize: Int,
    val images: ByteNdArray,
    val labels: ByteNdArray,
) : Iterator<ImageBatch> {
    val numImages = images.shape().size(0)
    var batchStart = 0

    override fun hasNext(): Boolean = batchStart < numImages

    override fun next(): ImageBatch {
        val nextBatchSize = min(batchSize.toLong(), numImages - batchStart)
        val range = range(batchStart.toLong(), batchStart + nextBatchSize)
        batchStart += nextBatchSize.toInt()
        return ImageBatch(images.slice(range), labels.slice(range))
    }

}
