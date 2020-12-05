#include "headers.hpp"

#define assert_message(expr, message) (static_cast<bool>(expr) ? void(0) : __assert_fail(message, __FILE__, __LINE__, __ASSERT_FUNCTION))

time_point<high_resolution_clock> start_epoch, end_epoch;
duration<double> time_elapsed;
#define tic start_epoch = high_resolution_clock::now();
#define toc                                 \
  end_epoch = high_resolution_clock::now(); \
  time_elapsed = end_epoch - start_epoch;

template <typename T>
ostream &operator<<(ostream &os, vector<T> &v) {
  for (T t : v) os << t << ' ';
  return os << '\n';
}

template <typename T>
istream &operator>>(istream &is, vector<T> &v) {
  for (T &t : v) is >> t;
  return is;
}

vector<bool> binary_representation(int n) {
  vector<bool> ans;
  while (n) {
    ans.push_back(n & 1);
    n >>= 1;
  }
  while (ans.size() != 32) {
    ans.push_back(0);
  }
  // reverse(ans.begin(), ans.end());
  return ans;
}

vector<pair<int, int>> sudoku_graph(int rank) {
  int sqrt_n = rank;
  int n = sqrt_n * sqrt_n;
  auto idx = [&](int r, int c) -> int {
    assert(r < n && c < n);
    return r * n + c;
  };
  vector<vector<bool>> adjacency_matrix(n * n, vector<bool>(n * n, 0));
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      int index = idx(row, col);
      for (int col_idx = 0; col_idx < n; col_idx++) {
        int neighbour_index = idx(row, col_idx);
        if (index < neighbour_index) {
          adjacency_matrix[index][neighbour_index] = true;
        }
      }

      for (int row_idx = 0; row_idx < n; row_idx++) {
        int neighbour_index = idx(row_idx, col);
        if (index < neighbour_index) {
          adjacency_matrix[index][neighbour_index] = true;
        }
      }

      int box_row_base = row / sqrt_n, box_col_base = col / sqrt_n;

      for (int box_row = 0; box_row < sqrt_n; box_row++) {
        for (int box_col = 0; box_col < sqrt_n; box_col++) {
          int i = box_row + box_row_base * sqrt_n, j = box_col_base * sqrt_n + box_col;
          int neighbour_index = idx(i, j);
          if (index < neighbour_index) {
            adjacency_matrix[index][neighbour_index] = true;
          }
        }
      }
    }
  }

  vector<pair<int, int>> constraints;
  for (int u = 0; u < n * n; u++) {
    for (int v = 0; v < n * n; v++) {
      if (adjacency_matrix[u][v]) {
        constraints.push_back({u + 1, v + 1});
      }
    }
  }

  return constraints;
}