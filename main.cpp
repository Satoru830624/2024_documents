#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <thread>

using Mats = std::vector<cv::Mat>;
using Points = std::vector<cv::Point>;

// B-Snakeの初期化
void initializeControlPoints(Points& controlPoints, const cv::Point& start, const cv::Point& end) {
	controlPoints.clear();
	controlPoints.push_back(start);
	controlPoints.push_back((2 * start + end) / 3);
	controlPoints.push_back((start + 2 * end) / 3);
	controlPoints.push_back(end);
}

// GVFの計算
cv::Mat calculateExternalForces(const cv::Mat& edgeImage) {
	// エッジの勾配を計算
	cv::Mat gradientX, gradientY;
	cv::Sobel(edgeImage, gradientX, CV_64F, 1, 0, 3);
	cv::Sobel(edgeImage, gradientY, CV_64F, 0, 1, 3);

	// GVFの計算
	cv::Mat gvfX = gradientX.clone();
	cv::Mat gvfY = gradientY.clone();
	double mu = 0.2; // GVFの平滑化パラメータ
	int iterations = 50;

	for (int iter = 0; iter < iterations; ++iter) {
		cv::Mat laplacianX, laplacianY;
		cv::Laplacian(gvfX, laplacianX, CV_64F, 3);
		cv::Laplacian(gvfY, laplacianY, CV_64F, 3);

		gvfX += mu * laplacianX - (gvfX - gradientX).mul(gradientX.mul(gradientX) + gradientY.mul(gradientY));
		gvfY += mu * laplacianY - (gvfY - gradientY).mul(gradientX.mul(gradientX) + gradientY.mul(gradientY));
	}

	// 結果を1つのマトリックスに統合
	std::vector<cv::Mat> gvfs = { gvfX, gvfY };
	cv::Mat externalForces;
	cv::merge(gvfs, externalForces);
	return externalForces;
}

// B-Snakeのパラメータを更新
void updateBSnakeParameters(Points& controlPoints, const cv::Mat& externalForces, double stepSize) {
	for (auto& point : controlPoints) {
		int x = std::min(std::max(point.x, 0), externalForces.cols - 1);
		int y = std::min(std::max(point.y, 0), externalForces.rows - 1);
		cv::Vec2d force = externalForces.at<cv::Vec2d>(y, x);

		point.x += stepSize * force[0];
		point.y += stepSize * force[1];
	}
}

// B-Snakeを描画
void drawBSnake(cv::Mat& image, const Points& controlPoints, const cv::Scalar& color) {
	for (size_t i = 0; i < controlPoints.size() - 1; ++i) {
		cv::line(image, controlPoints[i], controlPoints[i + 1], color, 2);
	}
}

// 画像のリサイズと表示
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

// 画像をファイルに保存
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
	std::filesystem::create_directories(output_folder); // 出力フォルダの作成

	auto img = cv::imread(img_path.string().c_str(), cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Image not found" << std::endl;
		return -1;
	}

	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

	// ガウシアンブラーを適用
	cv::Mat blurred_img;
	cv::GaussianBlur(gray_img, blurred_img, cv::Size(5, 5), 0);

	// エッジ検出
	cv::Mat edges;
	cv::Canny(blurred_img, edges, 50, 150); // しきい値を調整

	// エッジ検出結果の表示と保存（縮小）
	showResizedImage("Edges", edges, 0.5); // 50%縮小表示
	saveImage((output_folder / "edges.png").string(), edges);
	cv::waitKey(100);

	// マスク処理
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

	// マスク適用後のエッジ結果の表示と保存（縮小）
	showResizedImage("Masked Edges", edges, 0.5); // 50%縮小表示
	saveImage((output_folder / "masked_edges.png").string(), edges);
	cv::waitKey(100);

	// 初期コントロールポイントの設定（CHEVPアルゴリズムの結果を仮定）
	Points controlPoints;
	cv::Point start(100, 100); // 仮のスタートポイント
	cv::Point end(500, 500);   // 仮のエンドポイント
	initializeControlPoints(controlPoints, start, end);

	// 外力の計算
	cv::Mat externalForces = calculateExternalForces(edges);

	// 外力の表示と保存（縮小）
	if (!externalForces.empty()) {
		// X方向とY方向の成分を分離
		std::vector<cv::Mat> channels(2);
		cv::split(externalForces, channels);
		cv::Mat magnitude;
		cv::magnitude(channels[0], channels[1], magnitude);

		cv::Mat externalForcesMag;
		cv::normalize(magnitude, externalForcesMag, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		showResizedImage("External Forces", externalForcesMag, 0.5); // 50%縮小表示
		saveImage((output_folder / "external_forces.png").string(), externalForcesMag);
		cv::waitKey(100);
	}
	else {
		std::cerr << "External forces calculation failed." << std::endl;
		return -1;
	}

	// B-Snakeのパラメータ更新
	double stepSize = 0.1; // ステップサイズ
	for (int i = 0; i < 100; ++i) { // 100回反復
		updateBSnakeParameters(controlPoints, externalForces, stepSize);
	}

	// 結果の描画
	cv::Mat drawing_debug = img.clone();
	drawBSnake(drawing_debug, controlPoints, cv::Scalar(0, 255, 0));

	// 結果の画像を出力フォルダに保存
	std::string output_path = (output_folder / "fitted_result.png").string();
	saveImage(output_path, drawing_debug);

	// 結果の表示（縮小）
	showResizedImage("Fitted B-Snake", drawing_debug, 0.5); // 50%縮小表示
	cv::waitKey();

	return 0;
}
