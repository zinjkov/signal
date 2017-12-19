#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

const double	PI = 2. * acos(0),
                M = 300., // кол-во членов в интегральной сумме
                l = 12., // высота(м)
                N = 750., // частота дискритизации(сколько раз принемаем)
                mu = 0.018, // кооф ослабления
                length_t = 0.00001, // длитильность импульса (дельта, треугольник)
                delta_t = 0.4, // частота зондирования(c)
                V = 1., //скорость движения аппарата
                c = 1500., // скорость звука в воде(м/с)
                alpha1 = (179 * PI) / 180., // ширина диаграммы направленности
                alpha2 = (80 * PI) / 180.,
                beta = (1 * PI) / 180., // ширина диаграммы направленности в плоскости k_2 = 0
                sigma = mu*0.1;
double R = 20;

const int	H = 40 / (delta_t*V); // дистанция, которую прошел аппарат (м)
const int Nn = 750;
double norm_sine = 0;

fstream filesin("/home/zinjkov/qt-project/eugen_signal/sin.txt");
fstream filek("home/zinjkov/qt-project/eugen_signal/k3.txt");


double max(double a, double b)
{
    if (a <= b) return b;
    else return a;
}
double ifnan(double x)
{
    return (isnan(x) || isinf(x)) ? 0 : x;
}
double sqr(double a)
{
    return a*a;
}
double l_i(double t, double t_j)
{
    return (c*(t - t_j) / 2);
}
double mod_Vt(double k_2, double t_, double t)
{
    return ((t - t_) * (1 - sqr(V) / sqr(c)) / 2)*(c + (V*k_2) / (1 - (V * k_2) / c));
}
double mod_yx(double k_2, double t_, double t)
{
    return ((t - t_) * (1 - sqr(V) / sqr(c)) / 2)*(c - (V*k_2) / (1 - (V * k_2) / c));
}

double sigma_d(double y1, double y2)
{
    if (sqrt(sqr(y1 - 100) + sqr(y2 - 20)) < 5)
        return 0.3;
    if (sqrt(sqr(y1 - 200) + sqr(y2 - 20)) < 10)
        return 0.2;
    return 0.1;
}
double sigma_d(double t, double t_, double k_2)
{
    double y2 = V*t - k_2*mod_Vt(k_2, t_, t);
    double y1 = ifnan(sqrt(sqr((c*(t - t_) / 2)*(1 - sqr(V) / sqr(c)) + (V / c)*(V*t - y2)) - sqr(V*t - y2) - sqr(l)));
    return sigma_d(y1, y2);
}
double y1(double k_2, double t_, double t)
{
    double p = mod_Vt(k_2, t_, t);
    return sqrt(sqr(p) - sqr(k_2*p) - sqr(l));
}
double z1(double k_2, double k_3, double t_, double t)
{
    double Vt = mod_Vt(k_2, t_, t);
    return Vt*sqrt(1 - sqr(k_2) - sqr(k_3));
}
double yakobian_g(double k_2, double t_, double t)
{
    double Vt = mod_Vt(k_2, t_, t);
    double yx = mod_yx(k_2, t_, t);
    return (y1(k_2, t_, t) / (c*Vt))*((1 / Vt) - (k_2*V*(t - t_) / (Vt*yx)) + (1 / yx));
}
double yakobian_G(double k_2, double k_3, double t_, double t)
{
    double Vt = mod_Vt(k_2, t_, t);
    double yx = mod_yx(k_2, t_, t);
    return (z1(k_2, k_3, t_, t) / (c*sqr(Vt)))*(1. / Vt + 1. / yx + (k_2*(V*t - V*t_)) / (Vt*yx)*(-1. + sqr(k_3)));
}
double har(double x, double a, double b)
{
    return ((x >= a) && (x <= b));
}


void create_cv_img(cv::Mat frame)
{
    cv::imwrite("3.bmp", frame);
    cv::imshow("pict", frame);
    cv::waitKey(0);
}

bool write(char* f_name, double sigma_min, double sigma_max, double** sigma_d)
{
    FILE *f_out;
    cv::Mat frame(H, N, CV_8UC3);
//    fopen_s(&f_out, f_name, "wb");
    f_out = fopen(f_name, "wb");
    if (!f_out)
        return 0;

    char value = 0;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (sigma_d[i][j] <= sigma_min)
                value = 0;
            else if (sigma_d[i][j] >= sigma_max)
                value = 255;
            else value = char((sigma_d[i][j] - sigma_min) / (sigma_max - sigma_min)*255.);
            frame.at<cv::Vec3b>(i,j)[0] = value;
            frame.at<cv::Vec3b>(i,j)[1] = value;
            frame.at<cv::Vec3b>(i,j)[2] = value;
            fwrite(&value, sizeof(char), 1, f_out);
        }
    }

    fclose(f_out);
    create_cv_img(frame);

    return true;
}

inline double factorial(double x) {
    return tgamma(x+1);
}

double Si(double x, int eps = 0.e-20) {
    int i = 0;
    double ai = x;
    double answer = ai;

    while(abs(ai) > 0.e-20) {
        ai *= (pow(x, 2) * pow(-1, 1) * (2 * i + 1)) / ((2 * i + 2) * pow(2 * i + 3, 1));
        answer += ai;
        i++;
    }
    return answer;
}

double sine(double x, int n = 10) {
    int i = 0;
    double ai = x;
    double answer = ai;

    while(abs(ai) > 0.e-40) {
        ai *= (pow(x, 2) * pow(-1, 1) * (2 * i + 1)) / ((2 * i + 2) * pow(2 * i + 3, 2));
        answer += ai;
        i++;
    }
    return answer;
}

double integral_sine(double a, double b, double (*pf)(double, int)) {
    double res = 0;
    if (res == 0) {
        int n = 10000;
        double step = (b - a) / n;
        double point = a;
        for (int i = 0; i < n; ++i) {
            res += fabs(pf(point*7, 10));
            point += step;
        }
        res *= step;
    }
    return res;
}

double I_g(double t, double t_j)
{
    double _exp, delta;
    double k_3;
    double sum;
    double k_2 = 0., t_ = 0.;

    sum = 0.;

    for (int j = 0; j < M; j++)
    {
        double Vt;
        double yx;

        t_ = ((rand() / (double)RAND_MAX)*(-length_t) + length_t / 2. + t_j);

//        k_3 = (2.*l*c) / ((sqr(c) - sqr(V))*(t - t_));
        k_3 = - l / (mod_Vt(k_2, t_, t));
        delta = sin(beta) * ifnan(sqrt(1 - sqr(k_3)));

        k_2 = ((rand() / (double)RAND_MAX)*(-2 * delta) + delta);

        Vt = mod_Vt(k_2, t_, t);
        yx = mod_yx(k_2, t_, t);
        _exp = har(k_3, cos(alpha1), cos(alpha2));

        _exp *= (exp(-mu * (Vt + yx)));

        _exp *= (l / (pow(Vt, 3) * sqr(yx) *  abs(yakobian_g(k_2, t_, t))));

        _exp *= l / yx;

        //double tetha = acos(k_3);
        double A = norm_sine;
        double w = k_3 + 0.5;
        w *= R * 30;

//        cout << k_3 << endl;

//        cout << w << endl;
        double foo = 0;

        foo = fabs(Si(w));
//        cout << foo << endl;
        _exp *= foo;

        sum += (_exp * sigma_d(t, t_, k_2)) / PI;// / sqrt(1 - sqr(k_3));
    }

    return sum / M;
}

double I_G(double t, double t_j)
{
    double _exp, delta;
    double sum;
    double k_3, k_2, t_;

    sum = 0.;
    double sum2 = 0;

    for (int j = 0; j < M; j++)
    {
        double Vt;
        double yx;

        t_ = ((rand() / (double)RAND_MAX)*(-length_t) + length_t / 2. + t_j);
        k_3 = ((rand() / (double)RAND_MAX)*(cos(alpha2) + l / l_i(t, t_j))  -l / l_i(t, t_j));
        delta = sin(beta) * ifnan(sqrt(1 - sqr(k_3)));
        k_2 = ((rand() / (double)RAND_MAX)*(-2 * delta) + delta);



        if ((t - t_) >= 2.*l / c)
        {
            Vt = mod_Vt(k_2, t_, t);
            yx = mod_yx(k_2, t_, t);


            _exp = ((exp(-mu * (Vt + yx))) / (pow(Vt, 2) * sqr(yx) *  abs(yakobian_G(k_2, k_3, t_, t))));

            _exp *= cos(alpha2) + l / l_i(t, t_j);

            sum += (_exp * sigma) / (4*PI);

        }
    }
    return sum / M;
}
double A(double t, double t_j)
{
    if (-l > l*cos(alpha1)) return alpha1 - alpha2;
    else  return -alpha2 + acos(-l / l_i(t, t_j));
}

int main()
{
    norm_sine = sine(M_PI) - sine(-M_PI);
    cout.precision(20);
    cout << "integral: " << sine(M_PI) << endl;
    srand(time(NULL));
    double t = 1.;
    cout << norm_sine << endl;
    double sigma_min = 0, sigma_max = 0;

    double t_j, tmp;

    double** mas = (double**)malloc(sizeof(double) * H);
    for (int i = 0; i < H; i++)	mas[i] = (double*)malloc(sizeof(double) * N);


    double** clean = (double**)malloc(sizeof(double) * H);
    for (int i = 0; i < H; i++)	clean[i] = (double*)malloc(sizeof(double) * N);

    double** diff = (double**)malloc(sizeof(double) * H);
    for (int i = 0; i < H; i++)	diff[i] = (double*)malloc(sizeof(double) * N);


    ofstream myfile;
    myfile.precision(20);
    myfile.open("text.txt");

    double y_2, y_1;
    for (int j = 0; j < H; j++)
    {

        tmp = 0;
        for (int i = 0; i < N; i++)
        {

            t_j = delta_t * j;
            t = t_j + i * delta_t / N + (2.*l) / c;
            y_1 = sqrt(sqr(c * (t - t_j) / 2) - sqr(l));
            y_2 = V*t_j;
            clean[j][i] = sigma_d(y_1,y_2);

            t_j = delta_t * j;
            t = t_j + (i + 1) * delta_t / N + (2.*l) / c + length_t;
            y_1 = sqrt(sqr(l_i(t, t_j)) - sqr(l));
            y_2 = V*t_j;
            tmp = (2. * PI * pow(l_i(t, t_j), 4) * y_1 * exp(2.*mu*l_i(t, t_j))) / (c*sqr(l));


//            if((j == 0)) {
//                cout << 2 * l/(c*(t-t_j)) << endl;
//            }

            double sgm = sigma*c*exp(-2 * mu*l_i(t, t_j)) * A(t, t_j) / (8 * PI*sqr(l_i(t, t_j)));
            tmp *= I_g(t, t_j) + I_G(t, t_j);


            tmp -= sigma * sqr(l_i(t, t_j))*y_1 *A(t, t_j) / (4 * sqr(l));

            mas[j][i] = tmp;
            sigma_min = mas[0][0];
            if (sigma_max < tmp) sigma_max = tmp;
            if (sigma_min > tmp) sigma_min = tmp;

            double ddiff = fabs(mas[j][i] - clean[j][i]);
            diff[j][i] = ddiff;
//            if (sigma_max < ddiff) sigma_max = ddiff;
//            if (sigma_min > ddiff) sigma_min = ddiff;
        }
    }
    myfile.close();

    cout << sigma_max << "   " << sigma_min<< "\n";

    write("sigma_real_data.bin", 0, 0.3, mas);
    system("\"bin_array_to_image_bmp.exe\"");
    system("\"2.bmp\"");

//    system("pause");
    return 0;
}
