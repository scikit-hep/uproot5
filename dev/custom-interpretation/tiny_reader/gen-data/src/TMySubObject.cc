#include "TMySubObject.hh"
#include <string>

ClassImp(TMySubObject);

TMySubObject::TMySubObject() { set_data(); }
TMySubObject::TMySubObject(int &counter) { counter = set_data(counter); }

int TMySubObject::set_data(int counter) {
  // ------------ single elements ------------ //
  m_int = counter++;
  m_int16 = counter++;
  m_ulong = counter++;

  m_vec_int = {counter++, counter++, counter++};
  m_map_int_double = {{counter++, (double)counter++},
                      {counter++, (double)counter++}};
  m_stdstring = "I'm std::string " + std::to_string(counter++) + "!";
  for (int i = 0; i < 20; i++) {
    m_stdstring += ("I'm std::string " + std::to_string(counter++) + "!");
  }

  m_vec_vec_int = {{counter++, counter++, counter++}, {counter++, counter++}};
  m_vec_map_int_double = vector<map<int, double>>();
  for (int i = 0; i < 4; i++) {
    map<int, double> m;
    for (int j = 0; j < 3; j++) {
      m[counter++] = (double)counter++;
    }
    m_vec_map_int_double.push_back(m);
  }

  m_tstring = Form("I'm TString %d", counter++);

  m_tarrayf = TArrayF(3);
  m_tarrayf[0] = (float)counter++;
  m_tarrayf[1] = (float)counter++;
  m_tarrayf[2] = (float)counter++;

  // ------------ C-Arrays ------------ //
  for (int i = 0, counter = 29; i < 3; i++) {
    // 1d arrays
    m_carr_int[i] = counter++;
    m_carr_vec_int[i] = {counter++, counter++, counter++};
    m_carr_tstring[i] = Form("I'm TString %d", counter++);
    m_carr_tarrayf[i] = TArrayF(3);
    m_carr_tarrayf[i][0] = counter++;
    m_carr_tarrayf[i][1] = counter++;
    m_carr_tarrayf[i][2] = counter++;

    // 2d arrays
    for (int j = 0; j < 2; j++) {
      m_carr2d_int[j][i] = counter++;
      m_carr2d_vec_int[j][i] = {counter++, counter++, counter++};
      m_carr2d_tstring[j][i] = Form("I'm TString %d", counter++);
      m_carr2d_tarrayf[j][i] = TArrayF(3);
      m_carr2d_tarrayf[j][i][0] = counter++;
      m_carr2d_tarrayf[j][i][1] = counter++;
      m_carr2d_tarrayf[j][i][2] = counter++;
    }
  }

  return counter;
}
