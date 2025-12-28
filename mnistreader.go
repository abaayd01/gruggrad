package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"strings"
)

type MnistDataloader struct {
	trainingImagesFilepath string
	trainingLabelsFilepath string
	testImagesFilepath     string
	testLabelsFilepath     string
}

func NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels string) *MnistDataloader {
	return &MnistDataloader{
		trainingImagesFilepath: trainingImages,
		trainingLabelsFilepath: trainingLabels,
		testImagesFilepath:     testImages,
		testLabelsFilepath:     testLabels,
	}
}

func (m *MnistDataloader) ReadImagesLabels(imagesFilepath, labelsFilepath string) ([][]uint8, []uint8, error) {
	// Read labels
	labelsFile, err := os.Open(labelsFilepath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open labels file: %w", err)
	}
	defer labelsFile.Close()

	var labelsMagic, labelsSize uint32
	if err := binary.Read(labelsFile, binary.BigEndian, &labelsMagic); err != nil {
		return nil, nil, fmt.Errorf("failed to read labels magic: %w", err)
	}
	if labelsMagic != 2049 {
		return nil, nil, fmt.Errorf("magic number mismatch for labels, expected 2049, got %d", labelsMagic)
	}
	if err := binary.Read(labelsFile, binary.BigEndian, &labelsSize); err != nil {
		return nil, nil, fmt.Errorf("failed to read labels size: %w", err)
	}

	labels := make([]uint8, labelsSize)
	if _, err := labelsFile.Read(labels); err != nil {
		return nil, nil, fmt.Errorf("failed to read labels data: %w", err)
	}

	// Read images
	imagesFile, err := os.Open(imagesFilepath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open images file: %w", err)
	}
	defer imagesFile.Close()

	var imagesMagic, imagesSize, rows, cols uint32
	if err := binary.Read(imagesFile, binary.BigEndian, &imagesMagic); err != nil {
		return nil, nil, fmt.Errorf("failed to read images magic: %w", err)
	}
	if imagesMagic != 2051 {
		return nil, nil, fmt.Errorf("magic number mismatch for images, expected 2051, got %d", imagesMagic)
	}
	if err := binary.Read(imagesFile, binary.BigEndian, &imagesSize); err != nil {
		return nil, nil, fmt.Errorf("failed to read images size: %w", err)
	}
	if err := binary.Read(imagesFile, binary.BigEndian, &rows); err != nil {
		return nil, nil, fmt.Errorf("failed to read rows: %w", err)
	}
	if err := binary.Read(imagesFile, binary.BigEndian, &cols); err != nil {
		return nil, nil, fmt.Errorf("failed to read cols: %w", err)
	}

	imageData := make([]uint8, imagesSize*rows*cols)
	if _, err := imagesFile.Read(imageData); err != nil {
		return nil, nil, fmt.Errorf("failed to read image data: %w", err)
	}

	// Convert flat array to 2D array of images
	images := make([][]uint8, imagesSize)
	for i := uint32(0); i < imagesSize; i++ {
		images[i] = make([]uint8, rows*cols)
		copy(images[i], imageData[i*rows*cols:(i+1)*rows*cols])
	}

	return images, labels, nil
}

func (m *MnistDataloader) LoadData() ([][]uint8, []uint8, [][]uint8, []uint8, error) {
	xTrain, yTrain, err := m.ReadImagesLabels(m.trainingImagesFilepath, m.trainingLabelsFilepath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to load training data: %w", err)
	}

	xTest, yTest, err := m.ReadImagesLabels(m.testImagesFilepath, m.testLabelsFilepath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to load test data: %w", err)
	}

	return xTrain, yTrain, xTest, yTest, nil
}

func GenerateHTML(images [][]uint8, labels []uint8, outputPath string) error {
	var html strings.Builder

	html.WriteString(`<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MNIST Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .image-item {
            text-align: center;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        canvas {
            width: 140px;
            height: 140px;
            image-rendering: pixelated;
            border: 1px solid #ddd;
        }
        .label {
            margin-top: 8px;
            font-weight: bold;
            font-size: 16px;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>MNIST Dataset Viewer</h1>
    <div class="container">
`)

	for i := range images {
		html.WriteString(fmt.Sprintf(`        <div class="image-item">
            <canvas id="img-%d" width="28" height="28"></canvas>
            <div class="label">Label: %d</div>
        </div>
`, i, labels[i]))
	}

	html.WriteString(`    </div>
    <script>
`)

	for i, img := range images {
		html.WriteString(fmt.Sprintf(`        (function() {
            const canvas = document.getElementById('img-%d');
            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(28, 28);
            const pixels = [`, i))

		for j, pixel := range img {
			if j > 0 {
				html.WriteString(",")
			}
			html.WriteString(fmt.Sprintf("%d", pixel))
		}

		html.WriteString(`];
            for (let i = 0; i < pixels.length; i++) {
                imageData.data[i * 4] = pixels[i];
                imageData.data[i * 4 + 1] = pixels[i];
                imageData.data[i * 4 + 2] = pixels[i];
                imageData.data[i * 4 + 3] = 255;
            }
            ctx.putImageData(imageData, 0, 0);
        })();
`)
	}

	html.WriteString(`    </script>
</body>
</html>
`)

	return os.WriteFile(outputPath, []byte(html.String()), 0644)
}
