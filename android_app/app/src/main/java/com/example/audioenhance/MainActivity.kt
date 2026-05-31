package com.example.audioenhance

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.android.material.textfield.TextInputEditText
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private val TAG = "AudioEnhancer"
    private var module: Module? = null
    private var fullAudioData: FloatArray? = null
    private val SAMPLE_RATE = 16000
    private val CHUNK_SIZE = 32000 // 2 seconds

    private lateinit var btnSelect: Button
    private lateinit var btnPlayInput: Button
    private lateinit var btnEnhance: Button
    private lateinit var btnPlay: Button
    private lateinit var btnDownload: Button
    private lateinit var noisePrompt: TextInputEditText
    private lateinit var fileName: TextView
    private lateinit var progressBar: ProgressBar

    private val filePickerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val uri = result.data?.data
            uri?.let { 
                fileName.text = "File: ${it.path?.substringAfterLast("/")}"
                decodeAudio(it) 
            }
        }
    }

    private val saveFileLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        uri?.let { saveToUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnSelect = findViewById(R.id.btnSelect)
        btnPlayInput = findViewById(R.id.btnPlayInput)
        btnEnhance = findViewById(R.id.btnEnhance)
        btnPlay = findViewById(R.id.btnPlay)
        btnDownload = findViewById(R.id.btnDownload)
        noisePrompt = findViewById(R.id.noisePrompt)
        fileName = findViewById(R.id.fileName)
        progressBar = findViewById(R.id.progressBar)

        setupChips()
        checkPermissions()
        loadModel()

        btnSelect.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "audio/*"
            filePickerLauncher.launch(intent)
        }

        btnPlayInput.setOnClickListener { playCurrentInput() }

        btnEnhance.setOnClickListener {
            val prompt = noisePrompt.text.toString()
            if (prompt.isEmpty() || fullAudioData == null) {
                Toast.makeText(this, "Select audio and enter noise type", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (module == null) {
                Toast.makeText(this, "Model not loaded!", Toast.LENGTH_SHORT).show()
                loadModel()
                return@setOnClickListener
            }
            enhanceAudio(prompt)
        }

        btnPlay.setOnClickListener { playResult() }
        
        btnDownload.setOnClickListener {
            val prompt = noisePrompt.text.toString().filter { it.isLetterOrDigit() }.ifEmpty { "enhanced" }
            saveFileLauncher.launch("${prompt}_audio.wav")
        }
    }

    private fun setupChips() {
        // Noise types: dog, siren, rain, wind, engine, keyboard, baby, door and custom
        val chipIds = intArrayOf(R.id.chipDog, R.id.chipSiren, R.id.chipRain, R.id.chipWind, R.id.chipEngine, R.id.chipKeyboard, R.id.chipBaby, R.id.chipDoor, R.id.chipCustom)
        val names = arrayOf("Dog", "Siren", "Rain", "Wind", "Engine", "Keyboard", "Baby", "Door", "")
        
        for (i in chipIds.indices) {
            findViewById<Button>(chipIds[i]).setOnClickListener {
                noisePrompt.setText(names[i].lowercase())
                Toast.makeText(this, "Selected: ${names[i].ifEmpty { "Custom" }}", Toast.LENGTH_SHORT).show()
            }
        }

        // Mode cards (visual only)
        val modeIds = intArrayOf(R.id.modeVoice, R.id.modePodcast, R.id.modeMusic, R.id.modeField)
        for (id in modeIds) {
            findViewById<View>(id).setOnClickListener {
                Toast.makeText(this, "Optimizing mode...", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun loadModel() {
        Thread {
            try {
                val modelPath = assetFilePath("audio_enhancer_mobile.pt")
                module = Module.load(modelPath)
                runOnUiThread { Toast.makeText(this, "NeuroReady", Toast.LENGTH_SHORT).show() }
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, "Model Error: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }.start()
    }

    private fun checkPermissions() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.READ_MEDIA_AUDIO)
        } else {
            arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        if (ActivityCompat.checkSelfPermission(this, permissions[0]) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, permissions, 1)
        }
    }

    private fun decodeAudio(uri: Uri) {
        progressBar.visibility = View.VISIBLE
        btnEnhance.isEnabled = false
        btnPlayInput.isEnabled = false
        
        Thread {
            try {
                val extractor = MediaExtractor()
                contentResolver.openFileDescriptor(uri, "r")?.use { fd ->
                    extractor.setDataSource(fd.fileDescriptor)
                }

                val trackIndex = (0 until extractor.trackCount).firstOrNull {
                    extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
                } ?: throw Exception("No audio track found")

                extractor.selectTrack(trackIndex)
                val format = extractor.getTrackFormat(trackIndex)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
                
                val codec = MediaCodec.createDecoderByType(mime)
                codec.configure(format, null, null, 0)
                codec.start()

                val info = MediaCodec.BufferInfo()
                var isEOS = false
                val pcmData = mutableListOf<Short>()

                while (!isEOS) {
                    val inIndex = codec.dequeueInputBuffer(10000)
                    if (inIndex >= 0) {
                        val buffer = codec.getInputBuffer(inIndex)
                        val sampleSize = buffer?.let { extractor.readSampleData(it, 0) } ?: -1
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            isEOS = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }

                    var outIndex = codec.dequeueOutputBuffer(info, 10000)
                    while (outIndex >= 0) {
                        val buffer = codec.getOutputBuffer(outIndex)
                        buffer?.order(ByteOrder.LITTLE_ENDIAN)
                        while (buffer?.hasRemaining() == true) {
                            pcmData.add(buffer.short)
                        }
                        codec.releaseOutputBuffer(outIndex, false)
                        outIndex = codec.dequeueOutputBuffer(info, 0)
                    }
                }

                codec.stop()
                codec.release()
                extractor.release()

                fullAudioData = FloatArray(pcmData.size) { pcmData[it] / 32768.0f }
                
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnEnhance.isEnabled = true
                    btnPlayInput.isEnabled = true
                    Toast.makeText(this, "Loaded: ${pcmData.size / 16000}s", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    Toast.makeText(this, "Decoding failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun playCurrentInput() {
        val data = fullAudioData ?: return
        Thread {
            try {
                val pcm = ShortArray(data.size) { (data[it].coerceIn(-1.0f, 1.0f) * 32767).toInt().toShort() }
                val audioTrack = android.media.AudioTrack(
                    android.media.AudioManager.STREAM_MUSIC, SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    pcm.size * 2, android.media.AudioTrack.MODE_STATIC
                )
                audioTrack.write(pcm, 0, pcm.size)
                audioTrack.play()
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, "Play error: ${e.message}", Toast.LENGTH_SHORT).show() }
            }
        }.start()
    }

    private fun enhanceAudio(prompt: String) {
        if (module == null || fullAudioData == null) return
        progressBar.visibility = View.VISIBLE
        btnEnhance.isEnabled = false
        Toast.makeText(this, "Neural Processing...", Toast.LENGTH_SHORT).show()

        Thread {
            try {
                val NOISE_CLASSES = arrayOf("siren", "dog", "rain", "wind", "engine", "keyboard", "footsteps", "door", "baby", "clock")
                var tokenId = NOISE_CLASSES.indexOfFirst { prompt.lowercase().contains(it) }.coerceAtLeast(0)

                val inputData = fullAudioData!!
                val resultData = FloatArray(inputData.size)
                val tokenTensor = Tensor.fromBlob(longArrayOf(tokenId.toLong()), longArrayOf(1))

                var i = 0
                while (i < inputData.size) {
                    val end = minOf(i + CHUNK_SIZE, inputData.size)
                    val chunk = FloatArray(CHUNK_SIZE)
                    System.arraycopy(inputData, i, chunk, 0, end - i)
                    
                    val inputTensor = Tensor.fromBlob(chunk, longArrayOf(1, CHUNK_SIZE.toLong()))
                    val output = module!!.forward(IValue.from(inputTensor), IValue.from(tokenTensor)).toTensor()
                    val outChunk = output.dataAsFloatArray
                    
                    val copySize = minOf(outChunk.size, inputData.size - i)
                    System.arraycopy(outChunk, 0, resultData, i, copySize)
                    i += CHUNK_SIZE
                }

                saveToInternalWav(resultData, "restored.wav")

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnEnhance.isEnabled = true
                    btnPlay.isEnabled = true
                    btnDownload.isEnabled = true
                    Toast.makeText(this, "NeuroVoice Complete", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnEnhance.isEnabled = true
                    Toast.makeText(this, "AI Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun saveToInternalWav(data: FloatArray, fileName: String) {
        val file = File(filesDir, fileName)
        FileOutputStream(file).use { out ->
            writeWavHeader(out, 1, data.size, SAMPLE_RATE, 16)
            val buffer = ByteBuffer.allocate(data.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (f in data) {
                buffer.putShort((f.coerceIn(-1.0f, 1.0f) * 32767).toInt().toShort())
            }
            out.write(buffer.array())
        }
    }

    private fun saveToUri(uri: Uri) {
        try {
            val sourceFile = File(filesDir, "restored.wav")
            contentResolver.openOutputStream(uri)?.use { output ->
                sourceFile.inputStream().use { input ->
                    input.copyTo(output)
                }
            }
            Toast.makeText(this, "Exported Successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun writeWavHeader(out: FileOutputStream, channels: Int, totalSamples: Int, sampleRate: Int, bitDepth: Int) {
        val byteRate = sampleRate * channels * bitDepth / 8
        val totalDataLen = totalSamples * channels * bitDepth / 8
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(36 + totalDataLen)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1.toShort())
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort((channels * bitDepth / 8).toShort())
        header.putShort(bitDepth.toShort())
        header.put("data".toByteArray())
        header.putInt(totalDataLen)
        out.write(header.array())
    }

    private fun playResult() {
        val file = File(filesDir, "restored.wav")
        if (!file.exists()) return
        Thread {
            try {
                val bytes = file.readBytes().copyOfRange(44, file.length().toInt())
                val audioTrack = android.media.AudioTrack(
                    android.media.AudioManager.STREAM_MUSIC, SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    bytes.size, android.media.AudioTrack.MODE_STATIC
                )
                audioTrack.write(bytes, 0, bytes.size)
                audioTrack.play()
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, "Play error: ${e.message}", Toast.LENGTH_SHORT).show() }
            }
        }.start()
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }
        return file.absolutePath
    }
}
