package dataset

import org.tensorflow.ndarray.ByteNdArray
import org.tensorflow.ndarray.NdArrays
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.ndarray.index.Indices
import java.io.DataInputStream
import java.util.zip.GZIPInputStream

class MnistDataset private constructor(
    val trainingImages: ByteNdArray,
    val trainingLabels: ByteNdArray,
    val validationImages: ByteNdArray?,
    val validationLabels: ByteNdArray?,
    val testImages: ByteNdArray,
    val testLabels: ByteNdArray
) {
    fun trainingBatches(batchSize: Int): Iterable<ImageBatch> =
        object : Iterable<ImageBatch> {
            override fun iterator(): Iterator<ImageBatch> =
                ImageBatchIterator(batchSize, trainingImages, trainingLabels)
        }

    fun testBatches(batchSize: Int): Iterable<ImageBatch> =
        object : Iterable<ImageBatch> {
            override fun iterator(): Iterator<ImageBatch> =
                ImageBatchIterator(batchSize, testImages, testLabels)
        }

    fun numTestingExamples() = testLabels.shape().size(0)

    companion object {
        const val NUM_CLASSES = 10

        fun create(
            validationSize: Int,
            trainingImagesArchive: String,
            trainingLabelsArchive: String,
            testImageArchive: String,
            testLabelsArchive: String
        ): MnistDataset {
            val trainingImages = readArchive(trainingImagesArchive)
            val trainingLabels = readArchive(trainingLabelsArchive)
            val testImages = readArchive(testImageArchive)
            val testLabels = readArchive(testLabelsArchive)

            if (validationSize > 0) {
                return MnistDataset(
                    trainingImages.slice(Indices.sliceFrom(validationSize.toLong())),
                    trainingLabels.slice(Indices.sliceFrom(validationSize.toLong())),
                    trainingImages.slice(Indices.sliceTo(validationSize.toLong())),
                    trainingLabels.slice(Indices.sliceTo(validationSize.toLong())),
                    testImages,
                    testLabels
                )
            }
            return MnistDataset(trainingImages, trainingLabels, null, null, testImages, testLabels)
        }

        private fun readArchive(archiveName: String): ByteNdArray {
            val archiveStream = DataInputStream(
                GZIPInputStream(MnistDataset::class.java.classLoader.getResourceAsStream(archiveName))
            )
            archiveStream.readShort()
            archiveStream.readByte()
            // 識別子の一部分
            // 画像 -> 3, ラベル -> 1
            val numDims = archiveStream.readByte().toInt()
            val dimSizes = LongArray(numDims)

            // 次元数を決める作業
            // 画像: データ数 * 行数 * 列数
            // ラベル: ラベルデータの数
            var size = 1
            for (i in dimSizes.indices) {
                dimSizes[i] = archiveStream.readInt().toLong()
                size *= dimSizes[i].toInt()
            }

            // 次元数の要素文のデータを取るため、そのサイズのバイト配列を用意
            val bytes = ByteArray(size)
            archiveStream.readFully(bytes)
            // 次元(Shape)はdimSizesで取得した次元を利用
            // DataBuffersは取得したバイト列
            // これをByteNdArraysに変換して返す
            return NdArrays.wrap(Shape.of(*dimSizes), DataBuffers.of(bytes, true, false))
        }
    }
}
