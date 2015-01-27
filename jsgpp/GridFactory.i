// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at 
// sgpp.sparsegrids.org

%include "std_string.i"

%newobject sg::base::Grid::createLinearGrid(size_t dim);
%newobject sg::base::Grid::createLinearStretchedGrid(size_t dim);
%newobject sg::base::Grid::createLinearBoundaryGrid(size_t dim);
%newobject sg::base::Grid::createLinearTruncatedBoundaryGrid(size_t dim);
%newobject sg::base::Grid::createLinearTruncatedBoundaryGrid(BoudingBox& BB);
%newobject sg::base::Grid::createLinearStretchedTruncatedBoundaryGrid(size_t dim);
%newobject sg::base::Grid::createLinearStretchedTruncatedBoundaryGrid(Stretching& BB);
%newobject sg::base::Grid::createModLinearGrid(size_t dim);
%newobject sg::base::Grid::createPolyGrid(size_t dim, size_t degree);
%newobject sg::base::Grid::createModPolyGrid(size_t dim, size_t degree);
%newobject sg::base::Grid::createModWaveletGrid(size_t dim);
%newobject sg::base::Grid::createModBsplineGrid(size_t dim, size_t degree);
%newobject sg::base::Grid::createLinearGeneralizedTruncatedBoundaryGrid(size_t dim);
%newobject sg::base::Grid::createSquareRootGrid(size_t dim);
%newobject sg::base::Grid::createPrewaveletGrid(size_t dim);

%newobject sg::base::Grid::unserialize(std::string& istr);

%newobject sg::base::Grid::createGridGenerator();

%include "stl.i"
%include "typemaps.i"

%apply std::string *OUTPUT { std::string& ostr };
%apply std::string *INPUT { std::string& istr };

using namespace sg;
namespace sg
{
namespace base
{

class Grid
{
public:
	static Grid* createLinearGrid(size_t dim);
	static Grid* createLinearStretchedGrid(size_t dim);
	static Grid* createLinearBoundaryGrid(size_t dim);
	static Grid* createLinearTruncatedBoundaryGrid(size_t dim);
	static Grid* createLinearStretchedTruncatedBoundaryGrid(size_t dim);
	static Grid* createModLinearGrid(size_t dim);
	static Grid* createPolyGrid(size_t dim, size_t degree);
	static Grid* createModPolyGrid(size_t dim, size_t degree);
	static Grid* createModWaveletGrid(size_t dim);
	static Grid* createModBsplineGrid(size_t dim, size_t degree);
    static Grid* createSquareRootGrid(size_t dim);
	static Grid* createLinearGeneralizedTruncatedBoundaryGrid(size_t dim);
	static Grid* createPrewaveletGrid(size_t dim);
	
	static Grid* unserialize(const std::string& istr); 
	
protected:
	Grid();
	Grid(Grid& o);

public:
	virtual ~Grid();

public:	
	virtual sg::base::GridGenerator* createGridGenerator() = 0;	
	virtual sg::base::GridStorage* getStorage();
	virtual sg::base::BoundingBox* getBoundingBox();
	virtual sg::base::Stretching* getStretching();
	
	virtual void setBoundingBox(sg::base::BoundingBox& bb); 
    virtual void setStretching(sg::base::Stretching& bb); 

	virtual const char* getType() = 0;	
	virtual void serialize(std::string& ostr); 
	virtual std::string serialize();
	void refine(sg::base::DataVector* vector, int num);
	double eval(sg::base::DataVector& alpha, sg::base::DataVector& point);
	void insertPoint(size_t dim, unsigned int levels[], unsigned int indeces[], bool isLeaf);
	int getSize();
	
};
}
}

//these are just two new interfaces for consistency with Memento design pattern
%extend sg::base::Grid{
	sg::base::Grid* createMemento(){
		return $self;
	}
	
	static sg::base::Grid* setMemento(std::string& istr){
		return sg::base::Grid::unserialize(istr);
	}
};

	