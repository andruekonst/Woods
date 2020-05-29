# Woods: Decision Tree Ensembles

## TODO
* Implement Randomized Decision Tree
* Implement Median-split Decision Tree

## Installation

### Windows

1. Extract [boost](https://www.boost.org/users/download/) to `C:\libs\boost_1_73_0`
2. Run `C:\libs\boost_1_73_0\bootstrap.bat`
3. Run `.\b2 toolset=msvc-14.1 --address-model=64 --link=static --variant=debug --variant=release stage install`
3. Add environmental variables: ```
BOOST_INCLUDEDIR    C:\Boost\
// BOOST_LIBRARYDIR    C:\Boost\lib64-msvc-12.0
BOOST_ROOT          C:\Boost\boost
```