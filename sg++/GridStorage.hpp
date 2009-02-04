/*
This file is part of sg++, a program package making use of spatially adaptive sparse grids to solve numerical problems

Copyright (C) 2007  Jörg Blank (blankj@in.tum.de)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef GRIDSTORAGE_HPP
#define GRIDSTORAGE_HPP

#include "common/hash_map_config.hpp"

#include "exception/generation_exception.hpp"

#include "GridIndex.hpp"

#include <memory>
#include <string>
#include <sstream>
#include <exception>

#define SERIALIZATION_VERSION 1

namespace sg {

template<typename GIT>
class HashGridStorage;

/**
 * Main typedef for GridIndex
 */
typedef HashGridIndex<unsigned int, unsigned int> GridIndex;
/**
 * Main typedef for GridStorage
 */
typedef HashGridStorage<GridIndex> GridStorage;


/**
 * This class can be used for storage agnostic algorithms.
 * GridIndex has to support: constructor, get, set, push, rehash
 */
template<typename GIT>
class HashGridIterator
{
public:
	typedef GIT index_type;
	typedef typename GIT::index_type index_t;
	typedef typename GIT::level_type level_t;

	HashGridIterator(HashGridStorage<GIT>* storage) : storage(storage), index(storage->dim())
	{
		for(size_t i = 0; i < storage->dim(); i++)
		{
			index.push(i, 1, 1);
		}
		index.rehash();
		this->seq_ = storage->seq(&index);
	}


	/**
	 * left child in direction dim
	 */
	void left_child(size_t dim)
	{
		typename index_type::level_type l;
		typename index_type::index_type i;
		index.get(dim, l, i);
		index.set(dim, l + 1, 2 * i - 1);
		this->seq_ = storage->seq(&index);
	}

	/**
	 * right child in direction dim
	 */
	void right_child(size_t dim)
	{
		typename index_type::level_type l;
		typename index_type::index_type i;
		index.get(dim, l, i);
		index.set(dim, l + 1, 2 * i + 1);
		this->seq_ = storage->seq(&index);
	}

	/**
	 * resets the iterator to the top if dimension d
	 */
	void top(size_t d)
	{
		index.set(d, 1, 1);
		this->seq_ = storage->seq(&index);
	}

	/**
	 * hierarchical parent in direction dim
	 */
	void up(size_t d)
	{
		typename index_type::level_type l;
		typename index_type::index_type i;
		index.get(d, l, i);

		i /= 2;
		i += i % 2 == 0 ? 1 : 0;

		index.set(d, l - 1, i);
		this->seq_ = storage->seq(&index);
	}

	/**
	 * step right in direction dim
	 */
	 void step_right(size_t d)
	 {
		typename index_type::level_type l;
		typename index_type::index_type i;
		index.get(d, l, i);
		index.set(d, l, i + 2);
		this->seq_ = storage->seq(&index);

	 }

	/**
	 * returns true if there are no more childs in dimensioin d
	 */
	bool hint(size_t d) const
	{
		return false;
	}

	void get(size_t d, typename index_type::level_type &l, typename index_type::index_type &i) const
	{
		index.get(d, l, i);
	}

	void set(size_t d, typename index_type::level_type l, typename index_type::index_type i)
	{
		index.set(d, l, i);
	}

	void push(size_t d, typename index_type::level_type l, typename index_type::index_type i)
	{
		index.push(d, l, i);
	}

	/**
	 * returns the current sequence number
	 */
	size_t seq() const
	{
		return seq_;
	}


private:
	HashGridStorage<GIT>* storage;
	GIT index;
	size_t seq_;
};


/**
 * Generic hash table based index storage.
 */
template<typename GIT>
class HashGridStorage
{
public:
	typedef GIT index_type;
	typedef GIT* index_pointer;

    typedef std::hash_map<index_pointer, size_t, hash<index_pointer>, eqIndex<index_pointer> > grid_map;
    typedef typename grid_map::iterator grid_map_iterator;
    typedef typename grid_map::const_iterator grid_map_const_iterator;

	typedef std::vector<index_pointer> grid_list;
	typedef typename grid_list::iterator grid_list_iterator;

	typedef HashGridIterator<GIT> grid_iterator;

	HashGridStorage(size_t dim) : DIM(dim), list(), map()
	{
	}

	HashGridStorage(std::string& istr) : DIM(0), list(), map()
	{
    	std::istringstream istream;
    	istream.str(istr);

    	int version;
    	istream >> version;
    	if(version != SERIALIZATION_VERSION)
    	{
    		throw generation_exception("Unsupported version!");
    	}

    	istream >> DIM;

    	size_t num;
    	istream >> num;

    	for(size_t i = 0; i < num; i++)
    	{
    		index_pointer index = new GIT(istream);
    		list.push_back(index);
    		map[index] = i;
    	}
	}

	HashGridStorage(std::istream& istream) : DIM(0), list(), map()
	{
    	int version;
    	istream >> version;
    	if(version != SERIALIZATION_VERSION)
    	{
    		throw generation_exception("Unsupported version!");
    	}

    	istream >> DIM;

    	size_t num;
    	istream >> num;

    	for(size_t i = 0; i < num; i++)
    	{
    		index_pointer index = new GIT(istream);
    		list.push_back(index);
    		map[index] = i;
    	}
	}


	~HashGridStorage()
	{
		for(grid_list_iterator iter = list.begin(); iter != list.end(); iter++)
		{
			delete *iter;
		}
	}

	std::string serialize()
	{
		std::ostringstream ostream;
		this->serialize(ostream);
		return ostream.str();
	}

	void serialize(std::ostream& ostream)
	{
		ostream << SERIALIZATION_VERSION << " ";
		ostream << DIM << " ";
		ostream << list.size() << std::endl;

		for(grid_list_iterator iter = list.begin(); iter != list.end(); iter++)
		{
			(*iter)->serialize(ostream);
		}
	}

    void toString(std::ostream& stream)
    {
        stream << "[";
        int i = 0;
       	grid_map_iterator iter;
        for(iter = map.begin(); iter != map.end(); iter++, i++)
        {
            if(i != 0)
            {
                stream << ",";
            }
            stream << " ";
            iter->first->toString(stream);
            stream << " -> " << iter->second;
        }

        stream << " ]";
    }

    int size() const
    {
        return map.size();
    }

	int dim() const
	{
		return DIM;
	}

	unsigned int& operator[](index_pointer index)
	{
		return map[index];
	}

	index_pointer& operator[](size_t seq)
	{
		return list[seq];
	}

	GIT* get(size_t seq)
	{
		return list[seq];
	}


	size_t insert(index_type &index)
	{
		index_pointer insert = new GIT(&index);
		list.push_back(insert);
		return (map[insert] = this->seq()-1);
	}

	index_pointer create(index_type &index)
	{
		index_pointer insert = new GIT(&index);
		return insert;
	}

	void destroy(index_pointer index)
	{
		delete index;
	}

	unsigned int store(index_pointer index)
	{
		list.push_back(index);
		return (map[index] = this->seq() - 1);
	}

	grid_map_iterator find(index_pointer index)
	{
		return map.find(index);
	}

	grid_map_iterator begin()
	{
		return map.begin();
	}

	grid_map_iterator end()
	{
		return map.end();
	}

	/**
	 * Tests if index is in the storage
	 */
	bool has_key(GIT* index)
	{
		return map.find(index) != map.end();
	}

	/**
	 * Sets the index to the left child
	 */
	void left_child(GIT* index, size_t dim)
	{
		typename index_type::level_type l;
		typename index_type::index_type i;
		index->get(dim, l, i);
		index->set(dim, l + 1, 2 * i - 1);
	}

	/**
	 * Sets the index to the right child
	 */
	void right_child(GIT* index, size_t dim)
	{
		typename index_type::level_type l;
		typename index_type::index_type i;
		index->get(dim, l, i);
		index->set(dim, l + 1, 2 * i + 1);
	}

	/**
	 * Resets the index to the top level in direction d
	 */
	void top(GIT* index, size_t d)
	{
		index->set(d, 1, 1);
	}

	/**
	 * Gets the seq number for index
	 */
	size_t seq(GIT* index)
	{
		grid_map_iterator iter = map.find(index);
		if(iter != map.end())
		{
			return iter->second;
		}
		else
		{
			return map.size() + 1;
		}
	}

	/**
	 * Tests if seq number does not point to a valid grid index
	 */
	bool end(size_t s)
	{
		return s > map.size();
	}

	/**
	 * Should return true if there are no more childs in direction d
	 */
	bool hint(size_t d, GIT* index, size_t s)
	{
		return false;
	}

protected:
	/**
	 * returns the next sequence numbers
	 */
    size_t seq()
    {
        return list.size();
    }



private:
	size_t DIM;
	grid_list list;
    grid_map map;

};

}

#endif
