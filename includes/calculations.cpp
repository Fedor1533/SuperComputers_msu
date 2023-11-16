#include "calculations.h"

// operator== for struct Point
bool operator== (const Point& p1, const Point& p2)
{
    return (p1.x == p2.x) && (p1.y == p2.y);
}
// operator!= for struct Point
bool operator!= (const Point& p1, const Point& p2)
{
    return p1 != p2;
}
// operator << for struct Point
ostream& operator<<(ostream &os, const Point &p)
{
	os << "(" << p.x << ", " << p.y << ")" << ' ';
	return os;
}
// find length of vector
double length(const Point& p1, const Point& p2)
{
    double vec_len = pow(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2), 0.5);
    return vec_len;
}

// get area of polygon(vector of 2d points)
double polygon_area(vector<Point>& polygon)
{
    double area = 0;
	for (unsigned i = 0; i < polygon.size(); i++)
	{
        Point p1 = i ? polygon[i-1] : polygon[polygon.size() - 1];
        Point p2 = polygon[i];
		area += (p1.x - p2.x) * (p1.y + p2.y);
	}
	return abs(area) / 2;
}

//check if Point is in the area D: x^2 + 4y^2 < 1
bool point_in_area(const Point& p)
{
    return (pow(p.x, 2) + 4 * pow(p.y, 2) <= 1);
}

// check if polygon is inside the area D: x^2 + 4y^2 < 1
double polygon_in_area(vector<Point>& polygon)
{
    bool flag = true;
	for (auto point: polygon)
        flag = flag && point_in_area(point);
	return flag;
}

// check if polygon is outside the area D: x^2 + 4y^2 < 1
double polygon_out_area(vector<Point>& polygon)
{
	for (auto point: polygon)
        if (point_in_area(point)) return false;
	return true;
}

// function to find intersection of vectors and area boundary
Point find_intersect(const Point& p1, const Point& p2)
{
    // area boundary: x^2 + 4*y^2 < 1
    // expect that vector (p1, p2) parallel to OX or OY
    double inter_x, inter_y;
    if (p1.x == p2.x) {
        inter_y = pow((1. - pow(p1.x, 2)) / 4., 0.5);
        // y = +-((1 - x^2)/4)^(1/2)
        if ((inter_y < max(p1.y, p2.y)) && (inter_y > min(p1.y, p2.y))) {
            return Point(p1.x, inter_y);
        }
        else if ((-(inter_y) < max(p1.y, p2.y)) && (-(inter_y) > min(p1.y, p2.y))) {
            return Point(p1.x, -(inter_y));
        }
    }
    else if (p1.y == p2.y) {
        inter_x = pow((1. - 4 * pow(p1.y, 2)), 0.5);
        // x = +-(1 - 4 * y^2)^(1/2)
        if ((inter_x < max(p1.x, p2.x)) && (inter_x > min(p1.x, p2.x))) {
            return Point(inter_x, p1.y);
        }
        else if ((-(inter_x) < max(p1.x, p2.x)) && (-(inter_x) > min(p1.x, p2.x))) {
            return Point(-(inter_x), p1.y);
        }
    }
    else cout << "Didn't expect that (vector is not parallel to axis)" << endl;
    return Point(0, 0);
}

// get polygon P_ij for F_ij
vector<Point> get_Pij(Point& p, const double h1, const double h2)
{
    vector<Point> P_ij = {Point(p.x - h1/2, p.y - h2/2), Point(p.x - h1/2, p.y + h2/2), 
                          Point(p.x + h1/2, p.y + h2/2), Point(p.x + h1/2, p.y - h2/2),
                          Point(p.x - h1/2, p.y - h2/2)};
    return P_ij;
}

// get S_ij = part of P_ij intersected with D
vector<Point> get_Sij(vector<Point>& P_ij)
{
    // find intersection points with: (x)^2 + 4(y)^2 < 1 and add to new polygon if point inside D
    vector<Point> S_ij;
    for (size_t i = 1; i != P_ij.size(); ++i) {
        // if P_ij[i-1] inside D -> add to new polygon
        if (point_in_area(P_ij[i-1])) {S_ij.push_back(P_ij[i-1]);}

        Point inter_point = find_intersect(P_ij[i - 1], P_ij[i]);
        // no intersection -> Point(0, 0)
        if (inter_point.x != 0. || inter_point.y != 0.) 
        { 
            S_ij.push_back(inter_point);
        }
    }
    // add first point to 'close' polygon (if exists)
    if (S_ij.size() > 0) {S_ij.push_back(S_ij[0]);}
    return S_ij;
}

// get L_ij = part of vec_ij intersected with D
vector<Point> get_Lij(vector<Point>& vec_ij)
{
    // find intersection points with: (x)^2 + 4(y)^2 < 1 and return part of vector inside D
    Point inter_point = find_intersect(vec_ij[0], vec_ij[1]);
    vector<Point> l_ij;
    if (point_in_area(vec_ij[0])) { l_ij = {vec_ij[0], inter_point}; }
    else { l_ij = {inter_point, vec_ij[1]}; }
    return l_ij;
}