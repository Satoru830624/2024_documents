#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <thread>

using Mats = std::vector<cv::Mat>;
using Points = std::vector<cv::Point>;

// B-Snake�̏�����
void initializeControlPoints(Points& controlPoints, const cv::Point& start, const cv::Point& end) {
	controlPoints.clear();
	controlPoints.push_back(start);
	controlPoints.push_back((2 * start + end) / 3);
	controlPoints.push_back((start + 2 * end) / 3);
	controlPoints.push_back(end);
}

// GVF�̌v�Z
cv::Mat calculateExternalForces(const cv::Mat& edgeImage) {
	// �G�b�W�̌��z���v�Z
	cv::Mat gradientX, gradientY;
	cv::Sobel(edgeImage, gradientX, CV_64F, 1, 0, 3);
	cv::Sobel(edgeImage, gradientY, CV_64F, 0, 1, 3);

	// GVF�̌v�Z
	cv::Mat gvfX = gradientX.clone();
	cv::Mat gvfY = gradientY.clone();
	double mu = 0.2; // GVF�̕������p�����[�^
	int iterations = 50;

	for (int iter = 0; iter < iterations; ++iter) {
		cv::Mat laplacianX, laplacianY;
		cv::Laplacian(gvfX, laplacianX, CV_64F, 3);
		cv::Laplacian(gvfY, laplacianY, CV_64F, 3);

		gvfX += mu * laplacianX - (gvfX - gradientX).mul(gradientX.mul(gradientX) + gradientY.mul(gradientY));
		gvfY += mu * laplacianY - (gvfY - gradientY).mul(gradientX.mul(gradientX) + gradientY.mul(gradientY));
	}

	// ���ʂ�1�̃}�g���b�N�X�ɓ���
	std::vector<cv::Mat> gvfs = { gvfX, gvfY };
	cv::Mat externalForces;
	cv::merge(gvfs, externalForces);
	return externalForces;
}

// B-Snake�̃p�����[�^���X�V
void updateBSnakeParameters(Points& controlPoints, const cv::Mat& externalForces, double stepSize) {
	for (auto& point : controlPoints) {
		int x = std::min(std::max(point.x, 0), externalForces.cols - 1);
		int y = std::min(std::max(point.y, 0), externalForces.rows - 1);
		cv::Vec2d force = externalForces.at<cv::Vec2d>(y, x);

		point.x += stepSize * force[0];
		point.y += stepSize * force[1];
	}
}

// B-Snake��`��
void drawBSnake(cv::Mat& image, const Points& controlPoints, const cv::Scalar& color) {
	for (size_t i = 0; i < controlPoints.size() - 1; ++i) {
		cv::line(image, controlPoints[i], controlPoints[i + 1], color, 2);
	}
}

// �摜�̃��T�C�Y�ƕ\��
void showResizedImage(const std::string& windowName, const cv::Mat& image, double scale) {
	try {
		if (image.empty()) {
			throw std::runtime_error("Input image is empty");
		}

		cv::Mat resized;
		cv::resize(image, resized, cv::Size(), scale, scale);

		if (resized.empty()) {
			throw std::runtime_error("Resized image is empty");
		}

		cv::imshow(windowName, resized);
	}
	catch (const std::exception& e) {
		std::cerr << "Error in showResizedImage: " << e.what() << std::endl;
	}
}

// �摜���t�@�C���ɕۑ�
void saveImage(const std::string& filePath, const cv::Mat& image) {
	try {
		if (image.empty()) {
			throw std::runtime_error("Image is empty, cannot save to file.");
		}
		if (!cv::imwrite(filePath, image)) {
			throw std::runtime_error("Failed to save image to file: " + filePath);
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error in saveImage: " << e.what() << std::endl;
	}
}

int main(int argc, char* argv[]) {
	std::filesystem::path img_path("C:\\work\\Lane_detection\\img\\test.png");
	std::filesystem::path output_folder("C:\\work\\Lane_detection\\output\\");
	std::filesystem::create_directories(output_folder); // �o�̓t�H���_�̍쐬

	auto img = cv::imread(img_path.string().c_str(), cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Image not found" << std::endl;
		return -1;
	}

	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

	// �K�E�V�A���u���[��K�p
	cv::Mat blurred_img;
	cv::GaussianBlur(gray_img, blurred_img, cv::Size(5, 5), 0);

	// �G�b�W���o
	cv::Mat edges;
	cv::Canny(blurred_img, edges, 50, 150); // �������l�𒲐�

	// �G�b�W���o���ʂ̕\���ƕۑ��i�k���j
	showResizedImage("Edges", edges, 0.5); // 50%�k���\��
	saveImage((output_folder / "edges.png").string(), edges);
	cv::waitKey(100);

	// �}�X�N����
	std::vector<cv::Point> mask{
		cv::Point(0,0),
		cv::Point(edges.cols,0),
		cv::Point(edges.cols,1000),
		cv::Point(889,613),
		cv::Point(0,813)
	};
	const cv::Point* pts[1] = { mask.data() };
	int npt[1] = { static_cast<int>(mask.size()) };
	cv::fillPoly(edges, pts, npt, 1, CV_RGB(0, 0, 0));

	// �}�X�N�K�p��̃G�b�W���ʂ̕\���ƕۑ��i�k���j
	showResizedImage("Masked Edges", edges, 0.5); // 50%�k���\��
	saveImage((output_folder / "masked_edges.png").string(), edges);
	cv::waitKey(100);

	// �����R���g���[���|�C���g�̐ݒ�iCHEVP�A���S���Y���̌��ʂ�����j
	Points controlPoints;
	cv::Point start(100, 100); // ���̃X�^�[�g�|�C���g
	cv::Point end(500, 500);   // ���̃G���h�|�C���g
	initializeControlPoints(controlPoints, start, end);

	// �O�͂̌v�Z
	cv::Mat externalForces = calculateExternalForces(edges);

	// �O�͂̕\���ƕۑ��i�k���j
	if (!externalForces.empty()) {
		// X������Y�����̐����𕪗�
		std::vector<cv::Mat> channels(2);
		cv::split(externalForces, channels);
		cv::Mat magnitude;
		cv::magnitude(channels[0], channels[1], magnitude);

		cv::Mat externalForcesMag;
		cv::normalize(magnitude, externalForcesMag, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		showResizedImage("External Forces", externalForcesMag, 0.5); // 50%�k���\��
		saveImage((output_folder / "external_forces.png").string(), externalForcesMag);
		cv::waitKey(100);
	}
	else {
		std::cerr << "External forces calculation failed." << std::endl;
		return -1;
	}

	// B-Snake�̃p�����[�^�X�V
	double stepSize = 0.1; // �X�e�b�v�T�C�Y
	for (int i = 0; i < 100; ++i) { // 100�񔽕�
		updateBSnakeParameters(controlPoints, externalForces, stepSize);
	}

	// ���ʂ̕`��
	cv::Mat drawing_debug = img.clone();
	drawBSnake(drawing_debug, controlPoints, cv::Scalar(0, 255, 0));

	// ���ʂ̉摜���o�̓t�H���_�ɕۑ�
	std::string output_path = (output_folder / "fitted_result.png").string();
	saveImage(output_path, drawing_debug);

	// ���ʂ̕\���i�k���j
	showResizedImage("Fitted B-Snake", drawing_debug, 0.5); // 50%�k���\��
	cv::waitKey();

	return 0;
}
