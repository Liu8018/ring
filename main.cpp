#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

void ccomp(const cv::Mat &binImg, cv::Mat &ringImg)
{
    cv::Mat labels, stats, centroids;
    int nccomps = cv::connectedComponentsWithStats(binImg,
                                     labels,
                                     stats,     //nccomps×5的矩阵 表示每个连通区域的外接矩形和面积
                                     centroids);//nccomps×2的矩阵 表示每个连通区域的质心

    std::vector<int> colors(nccomps);
    colors[0] = 0; //背景保持黑色
    for(int i = 1; i < nccomps; i++ )
    {
        if(stats.at<int>(i,cv::CC_STAT_AREA) > binImg.rows*binImg.cols/20
           && stats.at<int>(i,cv::CC_STAT_AREA) < binImg.rows*binImg.cols/6)
            colors[i] = 255;
        else
            colors[i] = 0;
    }

    ringImg.create(binImg.size(),CV_8U);
    ringImg = cv::Scalar(0);
    for(int y=0;y<ringImg.rows;y++)
        for(int x=0;x<ringImg.cols;x++)
        {
            int label = labels.at<int>(y,x);
            ringImg.at<uchar>(y, x) = colors[label];
        }
}

void getContourArea(const std::vector<std::vector<cv::Point>> &contours,
                    std::vector<float> &contourAreas)
{
    contourAreas.clear();
    for(int i=0;i<contours.size();i++)
        contourAreas.push_back(cv::contourArea(contours[i]));
}

void findCircleContours(const std::vector<std::vector<cv::Point>> &contours,
                        const std::vector<float> &contourAreas,
                        std::vector<std::vector<cv::Point>> &circleContours,
                        std::vector<cv::Point> &centers,
                        std::vector<float> &radiuses,
                        std::vector<float> &circleContourAreas)
{
    circleContours.clear();
    for(int i=0;i<contours.size();i++)
    {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[i],center,radius);

        float circleArea = CV_PI*radius*radius;
        float contourArea = contourAreas[i];
        if(contourArea / circleArea > 0.9 && contourArea > 10000)
        {
            circleContours.push_back(contours[i]);
            centers.push_back(center);
            radiuses.push_back(radius);
            circleContourAreas.push_back(contourArea);
        }
    }
}

void orderContours(const std::vector<float> &contourAreas,
                   std::vector<std::vector<cv::Point>> &contours)
{
    for(int i=0;i<contours.size();i++)
    {
        int max_id=i;
        for(int j=i+1;j<contours.size();j++)
        {
            if(contourAreas[max_id] < contourAreas[j])
                max_id = j;
        }

        if(max_id == i)
            continue;
        else
        {
            std::vector<cv::Point> tmpContour;
            tmpContour.assign(contours[i].begin(),contours[i].end());
            contours[i].assign(contours[max_id].begin(),contours[max_id].end());
            contours[max_id].assign(tmpContour.begin(),tmpContour.end());
        }
    }
}

bool segment(const cv::Mat &src_img, cv::Mat &seg_img, int mode)
{
    bool debug_mode = false;

    //读入原图
    cv::Mat src;
    if(src_img.channels() == 3)
        cv::cvtColor(src_img,src,cv::COLOR_BGR2GRAY);

    //test
    if(debug_mode)
        cv::imshow("src",src);

    //自适应阈值
    cv::Mat adapt_img;
    cv::adaptiveThreshold(src,adapt_img,255,0,0,105,1);

    //test
    if(debug_mode)
        cv::imshow("adapt_img",adapt_img);

    //闭(开)运算
    cv::Mat morphed_img;
    if(mode == 0)
        cv::morphologyEx(adapt_img,morphed_img,cv::MORPH_CLOSE,cv::Mat(5,5,CV_8U,cv::Scalar(1)));
    else
        cv::morphologyEx(adapt_img,morphed_img,cv::MORPH_OPEN,cv::Mat(5,5,CV_8U,cv::Scalar(1)));

    //test
    if(debug_mode)
        cv::imshow("morphed_img",morphed_img);

    //筛选连通域
    ccomp(morphed_img,morphed_img);

    //test
    if(debug_mode)
        cv::imshow("color_img",morphed_img);

    //检测轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morphed_img,contours,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);
    std::vector<float> contourAreas;
    getContourArea(contours,contourAreas);

    //筛选轮廓
    std::vector<std::vector<cv::Point>> circleContours;
    std::vector<cv::Point> centers;
    std::vector<float> radiuses;
    std::vector<float> circleContourAreas;
    findCircleContours(contours,contourAreas,circleContours,centers,radiuses,circleContourAreas);
    if(circleContours.empty())
    {
        cv::waitKey();
        return false;
    }

    //按面积从大到小排序
    orderContours(circleContourAreas,circleContours);

    //test
    if(debug_mode)
    {
        cv::Mat circle_contour_testImg(src.size(),CV_8U,cv::Scalar(0));
        for(int i=0;i<circleContours.size();i++)
        {
            cv::drawContours(circle_contour_testImg,circleContours,i,cv::Scalar(255));
            cv::putText(circle_contour_testImg,std::to_string(i),circleContours[i][0],0,1,cv::Scalar(155),2);
        }
        cv::imshow("circle_contour_testImg",circle_contour_testImg);
    }

    //用圆拟合轮廓
    std::vector<cv::Point2f> circle_center(circleContours.size());
    std::vector<float> circle_radius(circleContours.size());
    for(int i=0;i<circleContours.size();i++)
        cv::minEnclosingCircle(circleContours[i],circle_center[i],circle_radius[i]);

    //绘制掩码
    cv::Mat mask(src.size(),CV_8U,cv::Scalar(0));
    cv::circle(mask,circle_center[0],circle_radius[0]*1.1,cv::Scalar(255),-1);
    cv::circle(mask,circle_center[0],circle_radius[0]*0.5,cv::Scalar(0),-1);

    //复制对应区域
    cv::Rect bdrect = cv::boundingRect(circleContours[0]);
    bdrect.x -= 50;
    bdrect.y -= 50;
    bdrect.width += 100;
    bdrect.height += 100;
    if(bdrect.x < 0) bdrect.x = 0;
    if(bdrect.y < 0) bdrect.y = 0;
    if(bdrect.x + bdrect.width > src_img.cols) bdrect.width = src_img.cols - bdrect.x;
    if(bdrect.y + bdrect.height > src_img.rows) bdrect.height = src_img.rows - bdrect.y;
    cv::Mat src_ROI = src_img(bdrect);
    cv::Mat mask_ROI = mask(bdrect);
    src_ROI.copyTo(seg_img, mask_ROI);

    //test
    if(debug_mode)
    {
        cv::imshow("seg_img",seg_img);
        if(cv::waitKey() == 'q')
            exit(0);
    }

    return true;
}

int main()
{
    int dataset_size = 126;

    for(int id=1;id<=dataset_size;id++)
    {
        std::string prefix = "/home/liu/Downloads/ring/ring_data/缺陷/斑点/bd";
        std::string inputFileName = prefix + std::to_string(id) + ".bmp";
        
        cv::Mat src_img = cv::imread(inputFileName);
        cv::Mat seg_img;
        if(!segment(src_img,seg_img,0))
        {
            std::cout<<"fail: "<<inputFileName<<std::endl;
            //exit(0);
        }
        
        std::string output_prefix = "/home/liu/Downloads/ring/ring_data/train/bd/";
        std::string outputFileName = output_prefix + std::to_string(id) + ".bmp";
        cv::imwrite(outputFileName,seg_img);
    }

    std::cout<<"finished!"<<std::endl;
    
    return 0;
}
