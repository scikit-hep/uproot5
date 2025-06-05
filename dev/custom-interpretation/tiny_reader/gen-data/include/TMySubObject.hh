#pragma once

#include <Rtypes.h>
#include <RtypesCore.h>
#include <TArrayF.h>
#include <TObject.h>
#include <TString.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::vector;

class TMySubObject : public TObject {
public:
  TMySubObject();
  TMySubObject(int &counter);

  int set_data(int counter = 0);

private:
  // ------------ single elements ------------ //
  // ctypes
  int m_int;
  int16_t m_int16;
  ULong_t m_ulong;

  // STL
  vector<int> m_vec_int;
  map<int, double> m_map_int_double;
  string m_stdstring;

  vector<vector<int>> m_vec_vec_int;
  vector<map<int, double>> m_vec_map_int_double;

  // ROOT types
  TString m_tstring;
  TArrayF m_tarrayf;

  // ------------ C-Arrays ------------ //
  // 1d arrays
  int m_carr_int[3];
  vector<int> m_carr_vec_int[3];
  TString m_carr_tstring[3];
  TArrayF m_carr_tarrayf[3];

  // 2d arrays
  int m_carr2d_int[2][3];
  vector<int> m_carr2d_vec_int[2][3];
  TString m_carr2d_tstring[2][3];
  TArrayF m_carr2d_tarrayf[2][3];

  ClassDef(TMySubObject, 1)
};
