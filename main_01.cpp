#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <file_io.hpp>
#include <img_edit.hpp>
#include <LaneDetection.h>	
#include <opencv2/opencv.hpp>

//�G�b�W�摜���w�肵���Z�N�V�����ɕ������A���ꂼ���sections�ɒǉ�
using Mats = std::vector<cv::Mat>;
using Points = std::vector<cv::Point>;

void MakeSectionImage(const cv::Mat& edge_image, Mats& sections, std::vector<int>& y_offsets)
{
	int num = 5;//5
	auto pix_height = edge_image.rows / num;
	for (auto i = 0; i < num ; i++)
	{
		cv::Rect crop(0, i*pix_height, edge_image.cols, pix_height);

		auto croped = edge_image(crop);
		sections.push_back(croped.clone());
		y_offsets.push_back(i * pix_height);
	}
}
struct LineParameter { 
	double a; 
	double b;
	double plot_y(double x) { return a * x + b; }
	double plot_x(double y) { return (y - b)/a; }

};
std::vector<cv::Point> GetMaskPollyline(const std::filesystem::path& ini_path,const cv::Mat& full_size_img)
{
	return std::vector<cv::Point>{ cv::Point(0,0),cv::Point(full_size_img.cols,0),cv::Point(full_size_img.cols, full_size_img.rows),
		cv::Point(930,170),cv::Point(930,110),cv::Point(710,140),cv::Point(430, full_size_img.rows),cv::Point(0, full_size_img.rows) };
}
//�摜�ɓK�p����}�X�N�̃|�����C���i���p�`�̒��_���X�g�j��Ԃ�
void MaskImage(const std::filesystem::path& path ,cv::Mat& img)
{
	std::vector<cv::Point> mask = GetMaskPollyline(path,img);
	const cv::Point* pts[1] = { mask.data() };
	int npt[1] = { (int)mask.size() };
	cv::fillPoly(img, pts, npt, 1, CV_RGB(0, 0, 0));
}
//�w�肳�ꂽ�t�H���_���̂��ׂẲ摜�ɑ΂��āA���[�����o�������s���A���̌��ʂ��o��
void EvalGraphContinue(const std::filesystem::path&src_folder)
{
	bool verbose_lm_detction = true;
	bool verbose_seed_gen = true;
	bool verbose_run_crf = true;
	bool verbose_validating = true;
	bool verbose = verbose_lm_detction | verbose_seed_gen | verbose_run_crf | verbose_validating;

	bool is_first{ true };
	LaneDetection ld;
	auto&& list = std::filesystem::directory_iterator(src_folder);
	int index = 0;
	for (const auto& img_path : list )
	{
		if (img_path.path().extension() != fs::path(".png"))
		{
			continue;
		}

		auto path = img_path.path().string();
		if (is_first) {
			ld.initialize_variable(path);
			is_first = false;
		}
		ld.initialize_Img(path);

		ld.lane_marking_detection(verbose_lm_detction);

		// supermarking generation and low-level association
		ld.seed_generation(verbose_seed_gen);

		auto&& start = std::chrono::high_resolution_clock::now();
		// CRF graph configuration & optimization using hungarian method
		ld.graph_generation(verbose_run_crf);
		auto&& end = std::chrono::high_resolution_clock::now();

		auto&& mark_num = ld.getMarkNum();

		auto&& processing_time =  std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		std::cout <<"graph gowing:" << processing_time.count()/1000.0 << "\t: graph num  "<< mark_num <<std::endl;
		
		// validating
		ld.validating_final_seeds(verbose_validating);

		if (verbose) {
			cv::waitKey(10);
		}
	}
	



}
//�摜�ɑ΂��ă��[�����o�������s���A���̌��ʂ�\��
void EvalGraph(const std::filesystem::path& img_path)
{
	bool verbose_lm_detction = true;
	bool verbose_seed_gen = true;
	bool verbose_run_crf = true;
	bool verbose_validating = true;
	bool verbose = verbose_lm_detction | verbose_seed_gen | verbose_run_crf | verbose_validating;


	LaneDetection ld;
	auto path = img_path.string();

	ld.initialize_variable(path);
	ld.initialize_Img(path);
	
	ld.lane_marking_detection(verbose_lm_detction);

	// supermarking generation and low-level association
	ld.seed_generation(verbose_seed_gen);

	// CRF graph configuration & optimization using hungarian method
	ld.graph_generation(verbose_run_crf);

	// validating
	ld.validating_final_seeds(verbose_validating);


	if (verbose) {
		cv::waitKey(10);
	}
	cv::waitKey();

}

void applyBSnake(const std::vector<cv::Point>& initialPoints, const cv::Mat& edgeImage, std::vector<cv::Point>& fittedPoints) {
	// �X�l�[�N�A���S���Y���̃p�����[�^
	int maxIterations = 200;
	double alpha = 0.1; // �A�����G�l���M�[�̏d��
	double beta = 0.1;  // �ȗ��G�l���M�[�̏d��
	double gamma = 2.0; // �C���[�W�G�l���M�[�̏d��

	fittedPoints = initialPoints;

	for (int iteration = 0; iteration < maxIterations; iteration++) {
		for (size_t i = 1; i < fittedPoints.size() - 1; i++) {
			cv::Point2f currentPoint = fittedPoints[i];

			// �����G�l���M�[�v�Z
			cv::Point2f previousPoint = fittedPoints[i - 1];
			cv::Point2f nextPoint = fittedPoints[i + 1];
			cv::Point2f continuity = alpha * (previousPoint + nextPoint - 2 * currentPoint);
			cv::Point2f curvature = beta * (nextPoint - 2 * currentPoint + previousPoint);

			// �摜�G�l���M�[�v�Z
			double minEnergy = std::numeric_limits<double>::max();
			cv::Point2f bestMove = currentPoint;
			for (int dx = -1; dx <= 1; dx++) {
				for (int dy = -1; dy <= 1; dy++) {
					cv::Point2f candidate = currentPoint + cv::Point2f(dx, dy);
					if (candidate.x >= 0 && candidate.x < edgeImage.cols && candidate.y >= 0 && candidate.y < edgeImage.rows) {
						double imageEnergy = -gamma * edgeImage.at<uchar>(candidate) / 255.0; // �G�b�W�̋��x�𐳋K�����Ďg�p
						double totalEnergy = imageEnergy + cv::norm(continuity + curvature);
						if (totalEnergy < minEnergy) {
							minEnergy = totalEnergy;
							bestMove = candidate;
						}
					}
				}
			}

			// �X�l�[�N�|�C���g�̍X�V
			if (bestMove != currentPoint) {
				fittedPoints[i] = bestMove;
			}
		}
	}
}

cv::Vec4f fitLine(const std::vector<cv::Point>& points) {
	cv::Vec4f line;
	cv::fitLine(points, line, cv::DIST_L2, 0, 0.01, 0.01);
	return line;
}

int main(int argc, char* argv[])
{
	std::filesystem::path img_path("C:\\work\\Lane_detection\\img\\test.png");
	std::filesystem::path output_folder("C:\\work\\Lane_detection\\output\\");
	std::filesystem::create_directories(output_folder); // �o�̓t�H���_�̍쐬

//	for (const auto& path : std::filesystem::directory_iterator("D:\\asaka\\vs_sim\\202406_line���m����\\image\\debug"))
//	{
////		EvalGraph(path.path());
//	}
	//EvalGraph(img_path);
	//EvalGraphContinue("G:\\2024_evals\\202406_line_utunomiya_png\\origin_300_8000");
	//return 0;
	//�摜�̓ǂݍ��݁��\��
	//namespace hstd = hi_ams::simulator::standard_io;
	auto&& img = cv::imread(img_path.string().c_str(), cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Image not found" << std::endl;
		return -1;
	}
	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
	//cv::imshow("src", img);
	//cv::Mat bit8_img = img.clone();
	//img.convertTo(bit8_img, CV_8UC1);

	// �K�E�V�A���u���[��K�p
	cv::Mat blurred_img;
	cv::GaussianBlur(gray_img, blurred_img, cv::Size(7, 7), 0);


	//�G�b�W���o
	cv::Mat edges;
	cv::Canny(gray_img, edges, 20, 160);//20, 50
	cv::imshow("edge", edges);
	cv::waitKey(100);
	//�}�X�N����
	std::vector<cv::Point> mask{ cv::Point(0,0),cv::Point(edges.cols,0), cv::Point(edges.cols,993), cv::Point(913,600), cv::Point(0,831), cv::Point(913,edges.rows), cv::Point(0,edges.rows) };
	//std::vector<cv::Point> mask{ cv::Point(0,0),cv::Point(edges.cols,0),cv::Point(edges.cols,edges.rows),
	//	cv::Point(930,170),cv::Point(930,110),cv::Point(710,140),cv::Point(430,edges.rows),cv::Point(0,edges.rows) };
	const cv::Point* pts[1] = { mask.data() };
	int npt[1] = { (int)mask.size() };
	cv::fillPoly(edges, pts,npt,1, CV_RGB(0,0,0));
	cv::imshow("edge_mask", edges);
	cv::waitKey(100);
	//�摜���Z�N�V�����ɕ���
	std::vector<cv::Mat> sections;
	std::vector<int> y_offsets;
	MakeSectionImage(edges,sections, y_offsets);
	auto drawing_debug = img.clone();

	std::vector<cv::Point> allFittedPoints; // �S�Z�N�V�����̃t�B�b�e�B���O�|�C���g��ێ�����x�N�g��

	//auto&& drawing_debug = edges.clone();
	//cv::cvtColor(drawing_debug, drawing_debug, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		int i = 0;
		for (const auto& section_image : sections)
		{
			//std::string win_name("section"); win_name += std::to_string(i);
			//cv::imshow(win_name.c_str(), section_image);



			//�n�t�ϊ�����
			std::vector<cv::Vec4i> lines;
			std::vector<LineParameter> line_params;
			const float  pi = 4*std::atan(1);
			cv::HoughLinesP(section_image, lines,1, pi / 180, 10, 20);//180,10,20
			// �n�t�ϊ����ʂ����������m�F���A���o���ꂽ���C����`��
			if (lines.empty()) {
				std::cerr << "No lines detected in section " << i << std::endl;
			}
			else {
				std::cout << "Detected " << lines.size() << " lines in section " << i << std::endl;
				for (const auto& line : lines) {
					cv::line(drawing_debug, cv::Point(line[0], line[1] + y_offsets[i]), cv::Point(line[2], line[3] + y_offsets[i]), cv::Scalar(0, 0, 255), 2);
				}
			}

			// �e�Z�N�V�����ɑ΂��ĈقȂ�F���g�p
			cv::Scalar color;
			switch (i) {
			case 0: color = CV_RGB(255, 0, 0); break;
			case 1: color = CV_RGB(0, 255, 0); break;
			case 2: color = CV_RGB(0, 0, 255); break;
			case 3: color = CV_RGB(255, 255, 0); break;
			case 4: color = CV_RGB(0, 255, 255); break;
			default: color = CV_RGB(255, 255, 255); break;
			}

			// ���o���ꂽ���C����B-Snake�Ńt�B�b�e�B���O
			for (const auto& line : lines)
			{
				std::vector<cv::Point> initialPoints = { cv::Point(line[0], line[1]), cv::Point(line[2], line[3]) };
				std::vector<cv::Point> fittedPoints;
				applyBSnake(initialPoints, section_image, fittedPoints);

				// �t�B�b�e�B���O���ʂ�`��
				for (size_t j = 0; j < fittedPoints.size() - 1; j++)
				{
					cv::line(drawing_debug, cv::Point(fittedPoints[j].x, fittedPoints[j].y + y_offsets[i]), cv::Point(fittedPoints[j + 1].x, fittedPoints[j + 1].y + y_offsets[i]), color, 2);
				}

				// �����|�C���g�̊m�F�p�`��
				cv::circle(drawing_debug, cv::Point(initialPoints[0].x, initialPoints[0].y + y_offsets[i]), 5, cv::Scalar(0, 255, 0), -1);
				cv::circle(drawing_debug, cv::Point(initialPoints[1].x, initialPoints[1].y + y_offsets[i]), 5, cv::Scalar(0, 255, 0), -1);
				// �t�B�b�e�B���O���ʂ�S�̂̃t�B�b�e�B���O�|�C���g�ɒǉ�
				for (const auto& pt : fittedPoints) {
					allFittedPoints.push_back(cv::Point(pt.x, pt.y + y_offsets[i]));
				}
				// �t�B�b�e�B���O�|�C���g�̊m�F�p�`��
				for (const auto& pt : fittedPoints) {
					cv::circle(drawing_debug, cv::Point(pt.x, pt.y + y_offsets[i]), 3, cv::Scalar(255, 0, 0), -1);
				}
			}

			//auto drawing_section = section_image.clone();
			//cv::cvtColor(drawing_section, drawing_section, cv::ColorConversionCodes::COLOR_GRAY2BGR);
			//int vanish_point = 64;
			//int vanish_offset = 0;
			//for (const auto & line : lines)
			//{
			//	cv::Point start{ line[0] ,line[1] };
			//	cv::Point end{ line[2] ,line[3] };
			//	//���o�������C���̕`��
			//	cv::line(drawing_section, start, end, CV_RGB(255, 0, 0),2);
			//	double a = (end.y -start.y) / (end.x -start.x + DBL_MIN);
			//	double b = start.y - a* start.x;
			//	LineParameter  line_plotter{ a, b };
			//	line_params.push_back(line_plotter);
			//	//�����_�̕`��
			//	cv::Point near_x_axis_point;
			//	if (start.x < end.x)
			//	{
			//		near_x_axis_point.x = start.x;
			//		near_x_axis_point.y = start.y;
			//	}
			//	else
			//	{
			//		near_x_axis_point.x = end.x;
			//		near_x_axis_point.y = end.y;
			//	}
			//	auto&& getpoint_from_x = [&line_plotter](auto x) { return cv::Point(x, (int)line_plotter.plot_y(x)); };
			//	auto&& getpoint_from_y = [&line_plotter](auto y) { return cv::Point((int)line_plotter.plot_x(y),y ); };
			//	int x1 = 0;
			//	int x2 = drawing_section.cols;
			//	int y2 = drawing_section.rows;
			//	cv::circle(drawing_section, cv::Point{ near_x_axis_point.x,near_x_axis_point.y }, 5, CV_RGB(0, 255, 0));
			//	cv::line(drawing_section, getpoint_from_x(0), getpoint_from_x(x2), CV_RGB(0, 255, 0));
			//	cv::line(drawing_section, getpoint_from_y(0), getpoint_from_y(y2), CV_RGB(0, 0, 255));
			//	cv::Point vanish_pos = getpoint_from_y(vanish_point - vanish_offset);
			//	vanish_pos.y += vanish_offset;
			//	cv::circle(drawing_debug, vanish_pos, 5, CV_RGB(0, 0, 255));

			//}

			//���ʂ̏o��
			//std::string det("hogh_result"); det += std::to_string(i);
			//cv::imshow(det.c_str(), drawing_section);
			//cv::imshow("debug_full", drawing_debug);
			//cv::waitKey(50);

			i++;
			//vanish_offset += drawing_section.rows;
		}

	// �S�̂̃t�B�b�e�B���O�|�C���g���g���Ē������t�B�b�e�B���O
	if (!allFittedPoints.empty()) {
		cv::Vec4f line = fitLine(allFittedPoints);
		float vx = line[0], vy = line[1];
		float x = line[2], y = line[3];

		int left_y = static_cast<int>((0 - x) * vy / vx + y);
		int right_y = static_cast<int>((drawing_debug.cols - x) * vy / vx + y);

		cv::line(drawing_debug, cv::Point(0, left_y), cv::Point(drawing_debug.cols, right_y), cv::Scalar(0, 255, 255), 2);
	}
	// ���ʂ̉摜���o�̓t�H���_�ɕۑ�
	std::string output_path = (output_folder / "fitted_result.png").string();
	cv::imwrite(output_path, drawing_debug);

	cv::imshow("Fitted B-Snake", drawing_debug);
	cv::waitKey();

	return 0;
}