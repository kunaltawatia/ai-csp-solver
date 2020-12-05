#include "headers.hpp"
#include "utility.hpp"
#define M 250  // max number of nodes in the graph, used in BITSET, lower is better for perf.

typedef unsigned short int usi;
class Sudoku {
  usi n, sqrt_n, full_domain;
  vector<vector<usi>> matrix, domain;
  priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> domain_heap;

 public:
  Sudoku(int n = 9) : n(n), sqrt_n(sqrt(n)), full_domain(((1 << n) - 1) << 1) {
    flush();
  }

  void flush() {
    matrix.clear();
    matrix.resize(n, vector<usi>(n, 0));
    domain.clear();
    domain.resize(n, vector<usi>(n, full_domain));
    while (domain_heap.size())
      domain_heap.pop();
    for (int row = 0; row < n; row++) {
      for (int column = 0; column < n; column++) {
        domain_heap.push({n, {row, column}});
      }
    }
  }

  void input() {
    cin >> matrix;
  }

  void initialise(vector<int> values) {
    int idx = 0;
    for (int row = 0; row < n; row++) {
      for (int col = 0; col < n; col++) {
        matrix[row][col] = values[idx++];
      }
    }
    assert(idx == values.size());
  }

  vector<int> get_values() {
    vector<int> ans;
    for (int idx = 0; idx < n * n; idx++) {
      ans.push_back(matrix[idx / n][idx % n]);
    }
    return ans;
  }

  void print() {
    for (int row = 0; row < n; row++) {
      for (int col = 0; col < n; col++) {
        cout << matrix[row][col] << " ";
      }
      cout << endl;
    }
  }

  void update(int row, int column, usi value) {
    assert(value <= n);
    if (value) {
      matrix[row][column] = value;
      domain[row][column] = 0;

      usi mark = 1 << value;

      for (int col_idx = 0; col_idx < n; col_idx++) {
        if (!matrix[row][col_idx]) {
          domain[row][col_idx] ^= (domain[row][col_idx] & mark);
          domain_heap.push({__builtin_popcount(domain[row][col_idx]), {row, col_idx}});
        }
      }

      for (int row_idx = 0; row_idx < n; row_idx++) {
        if (!matrix[row_idx][column]) {
          domain[row_idx][column] ^= (domain[row_idx][column] & mark);
          domain_heap.push({__builtin_popcount(domain[row_idx][column]), {row_idx, column}});
        }
      }

      int box_base_row = row / sqrt_n, box_base_col = column / sqrt_n;
      for (int box_idx = 0; box_idx < n; box_idx++) {
        int i = sqrt_n * box_base_row + box_idx / sqrt_n;
        int j = sqrt_n * box_base_col + box_idx % sqrt_n;
        if (!matrix[i][j]) {
          domain[i][j] ^= (domain[i][j] & mark);
          domain_heap.push({__builtin_popcount(domain[i][j]), {i, j}});
        }
      }
    } else {
      matrix[row][column] = 0;
      for (int col_idx = 0; col_idx < n; col_idx++) {
        if (!matrix[row][col_idx]) {
          domain[row][col_idx] = possible_assignment(row, col_idx);
          domain_heap.push({__builtin_popcount(domain[row][col_idx]), {row, col_idx}});
        }
      }

      for (int row_idx = 0; row_idx < n; row_idx++) {
        if (!matrix[row_idx][column]) {
          domain[row_idx][column] = possible_assignment(row_idx, column);
          domain_heap.push({__builtin_popcount(domain[row_idx][column]), {row_idx, column}});
        }
      }

      int box_base_row = row / sqrt_n, box_base_col = column / sqrt_n;
      for (int box_idx = 0; box_idx < n; box_idx++) {
        int i = sqrt_n * box_base_row + box_idx / sqrt_n;
        int j = sqrt_n * box_base_col + box_idx % sqrt_n;
        if (!matrix[i][j]) {
          domain[i][j] = possible_assignment(i, j);
          domain_heap.push({__builtin_popcount(domain[i][j]), {i, j}});
        }
      }

      domain[row][column] = possible_assignment(row, column);
      domain_heap.push({__builtin_popcount(domain[row][column]), {row, column}});
    }
  }

  int possible_assignment(int row, int column) {
    int row_bitmask = 0, col_bitmask = 0, box_bitmask = 0;
    for (int col_idx = 0; col_idx < n; col_idx++) {
      row_bitmask |= (1 << matrix[row][col_idx]);
    }
    for (int row_idx = 0; row_idx < n; row_idx++) {
      col_bitmask |= (1 << matrix[row_idx][column]);
    }
    int box_base_row = row / sqrt_n, box_base_col = column / sqrt_n;
    for (int box_idx = 0; box_idx < n; box_idx++) {
      int i = sqrt_n * box_base_row + box_idx / sqrt_n;
      int j = sqrt_n * box_base_col + box_idx % sqrt_n;
      box_bitmask |= (1 << matrix[i][j]);
    }

    int node_bitmask = ~(row_bitmask | col_bitmask | box_bitmask) & full_domain;

    return node_bitmask;
  }

  vector<usi> values_from_bitmask(int bitmask) {
    vector<usi> ans;
    for (usi value = 1; value <= n; value++) {
      if ((1 << value) & bitmask) {
        ans.push_back(value);
      }
    }
    return ans;
  }

  pair<int, int> most_conflicting_node() {
    int mn_domain = 0;
    pair<int, int> ans = {-1, -1};
    for (int row = 0; row < n; row++) {
      for (int column = 0; column < n; column++) {
        if (!matrix[row][column]) {
          int bitmask = possible_assignment(row, column);
          int current_domain_size = __builtin_popcount(bitmask);
          if (current_domain_size < mn_domain || (mn_domain == 0 && current_domain_size)) {
            mn_domain = current_domain_size;
            ans = {row, column};
          }
        }
      }
    }
    return ans;
  }

  pair<int, int> most_conflicting_node_optimized() {
    while (domain_heap.size()) {
      auto top = domain_heap.top();
      int domain_size = top.first, row = top.second.first, column = top.second.second;
      if (domain_size == __builtin_popcount(domain[row][column])) {
        return top.second;
      }
      domain_heap.pop();
    }
    return {-1, -1};
  }

  bool backtrack_with_ordering_optimized() {
    pair<int, int> node = most_conflicting_node_optimized();
    if (node.first == -1) {
      return valid();
    }
    int row = node.first, column = node.second;
    for (usi value : values_from_bitmask(domain[row][column])) {
      update(row, column, value);
      bool response = backtrack_with_ordering_optimized();
      if (response) return true;
    }
    update(row, column, 0);
    return false;
  }

  bool backtrack_with_ordering() {
    pair<int, int> node = most_conflicting_node();
    if (node.first == -1) {
      return valid();
    }
    int row = node.first, column = node.second;
    for (usi value : values_from_bitmask(possible_assignment(row, column))) {
      update(row, column, value);
      bool response = backtrack_with_ordering();
      if (response) return true;
    }
    update(row, column, 0);
    return false;
  }

  bool backtrack() {
    for (int row = 0; row < n; row++) {
      for (int column = 0; column < n; column++) {
        if (!matrix[row][column]) {
          for (usi value : values_from_bitmask(possible_assignment(row, column))) {
            update(row, column, value);
            bool response = backtrack();
            if (response) return true;
          }
          update(row, column, 0);
          return false;
        }
      }
    }
    return valid();
  }

  bool valid() {
    bool rows_valid = true, columns_valid = true, boxes_valid = true;

    // check for rows
    for (int row = 0; row < n && rows_valid; row++) {
      bool present[n + 1] = {};
      for (int column = 0; column < n; column++) {
        usi value = matrix[row][column];
        present[value] = 1;
      }
      for (usi value = 1; value <= n; value++) {
        if (!present[value]) rows_valid = false;
      }
    }

    // check for columns
    for (int column = 0; column < n && columns_valid; column++) {
      bool present[n + 1] = {};
      for (int row = 0; row < n; row++) {
        usi value = matrix[row][column];
        present[value] = 1;
      }
      for (usi value = 1; value <= n; value++) {
        if (!present[value]) columns_valid = false;
      }
    }

    // check for boxes
    for (int box_row = 0; box_row < sqrt_n && boxes_valid; box_row++) {
      for (int box_col = 0; box_col < sqrt_n && boxes_valid; box_col++) {
        bool present[n + 1] = {};

        for (int row = 0; row < sqrt_n; row++) {
          for (int col = 0; col < sqrt_n; col++) {
            usi value = matrix[box_row * sqrt_n + row][box_col * sqrt_n + col];
            present[value] = true;
          }
        }

        for (usi value = 1; value <= n; value++) {
          if (!present[value]) boxes_valid = false;
        }
      }
    }

    bool sudoku_valid = rows_valid && columns_valid && boxes_valid;
    return sudoku_valid;
  }

  long double last_epoch = 0;
  bool solve() {
    tic;
    bool response = backtrack();
    toc;
    last_epoch = time_elapsed.count();
    return response;
  }
};

class Graph {
 private:
  struct Variable {
    int value, index;
    stack<bitset<M>> domain_stack;  // later stack<bitset<M>>
    vector<Variable *> contacts;
    inline bool unassigned() { return value == 0; }
  };
  usi n, mx_value;
  bitset<M> full_domain;
  vector<Variable> nodes;

  vector<set<usi>> domain_groups;
  usi min_domain_size;

 public:
  Graph(int n, vector<pair<int, int>> constraints, int mx_value) : n(n), mx_value(mx_value) {
    full_domain = 0;
    for (int idx = 1; idx <= mx_value; idx++) {
      full_domain[idx] = 1;
    }

    nodes.resize(n + 1);                 // (+ 1) for indexing from 1, ...
    domain_groups.resize(mx_value + 1);  // (+ 1) for indexing from 1, ...
    for (auto constraint : constraints) {
      int u = constraint.first, v = constraint.second;
      nodes[u].contacts.push_back(&nodes[v]);
      nodes[v].contacts.push_back(&nodes[u]);
    }
    flush_values();
  };

  void flush_values() {
    int idx = 0;
    for (auto &node : nodes) {
      node.value = 0;
      node.index = idx++;
      while (!node.domain_stack.empty())
        node.domain_stack.pop();
      node.domain_stack.push(full_domain);
    }
    refresh_domain_groups();
  }

  void initialise(vector<int> values) {
    for (int idx = 1; idx <= n; idx++) {
      auto &node = nodes[idx];
      usi value = values[idx - 1];
      if (value) {
        node.value = value;
        bitset<M> mark(0);
        mark[value] = 1;
        for (auto neighbour : node.contacts) {
          bitset<M> prev_domain = neighbour->domain_stack.top();
          bitset<M> domain = prev_domain ^ (prev_domain & mark);
          neighbour->domain_stack.pop();
          neighbour->domain_stack.push(domain);
        }
      }
    }
    refresh_domain_groups();
    assert(satisfies_constraints());
  }

  void refresh_min_domain_size() {
    min_domain_size = 0;
    for (usi value = 0; value <= mx_value; value++) {
      if (domain_groups[value].size()) {
        min_domain_size = value;
        return;
      }
    }
  }

  void refresh_domain_groups() {
    for (usi value = 0; value <= mx_value; value++) {
      domain_groups[value].clear();
    }

    for (int idx = 1; idx <= n; idx++) {
      auto &node = nodes[idx];
      if (node.unassigned()) {
        usi domain_size = node.domain_stack.top().count();
        domain_groups[domain_size].insert(node.index);
      }
    }

    min_domain_size = 0;
    for (usi value = 0; value <= mx_value; value++) {
      if (domain_groups[value].size()) {
        min_domain_size = value;
        break;
      }
    }

    return;
  }

  void print() {
    for (int idx = 1; idx <= n; idx++) {
      Variable &node = nodes[idx];
      cout << idx << " " << node.value << "\n";
    }
  }

  void print_sudoku() {
    vector<int> assignment = get_values();
    for (int i = 0; i < sqrt(n); i++) {
      for (int j = 0; j < sqrt(n); j++) {
        cout << assignment[i * 9 + j] << " ";
      }
      cout << endl;
    }
  }

  vector<usi> values_from_bitmask(bitset<M> bitmask) {
    vector<usi> ans;
    for (usi value = 1; value <= mx_value; value++) {
      if (bitmask[value]) {
        ans.push_back(value);
      }
    }
    return ans;
  }

  void update(usi value, Variable *node) {
    node->value = value;
    bitset<M> mark(0);
    mark[value] = 1;
    bool domain_size_TBR = false;  // whther to refresh `min_domain_size`

    usi domain_size = node->domain_stack.top().count();
    if (value) {
      domain_groups[domain_size].erase(node->index);
      if (domain_size == min_domain_size && !domain_groups[min_domain_size].size()) {
        domain_size_TBR = true;
      }
    } else {
      domain_groups[domain_size].insert(node->index);
      min_domain_size = min(domain_size, min_domain_size);
    }

    for (auto neighbour : node->contacts) {
      if (neighbour->unassigned()) {
        bitset<M> prev_domain = neighbour->domain_stack.top(), domain;
        if (!value) {
          neighbour->domain_stack.pop();
          domain = neighbour->domain_stack.top();
        } else {
          domain = prev_domain ^ (prev_domain & mark);
          neighbour->domain_stack.push(domain);
        }

        usi prev_domain_size = prev_domain.count(), domain_size = domain.count();
        if (prev_domain_size != domain_size) {
          domain_groups[prev_domain_size].erase(neighbour->index);
          domain_groups[domain_size].insert(neighbour->index);
          if (prev_domain_size > domain_size) {
            min_domain_size = min(domain_size, min_domain_size);
          } else if (prev_domain_size == min_domain_size && !domain_groups[min_domain_size].size()) {
            domain_size_TBR = true;
          }
        }
      }
    }
    if (domain_size_TBR) refresh_min_domain_size();
  }

  void update_naive(usi value, Variable *node) {
    node->value = value;
    bitset<M> mark(0);
    mark[value] = 1;

    for (auto neighbour : node->contacts) {
      if (neighbour->unassigned()) {
        bitset<M> prev_domain = neighbour->domain_stack.top(), domain;
        if (!value) {
          neighbour->domain_stack.pop();
          for (auto next_neighbour : neighbour->contacts) {
            // stub operations here
            domain |= next_neighbour->domain_stack.top();
          }

          domain = neighbour->domain_stack.top();

        } else {
          domain = prev_domain ^ (prev_domain & mark);
          neighbour->domain_stack.push(domain);
        }
      }
    }
  }

  Variable *first_unassigned_variable() {
    Variable *ans = NULL;

    for (int idx = 1; idx <= n; idx++) {
      Variable &node = nodes[idx];
      if (node.unassigned()) {
        ans = &node;
        break;
      }
    }
    return ans;
  }

  Variable *most_constrained_variable() {
    Variable *ans = NULL;
    if (!min_domain_size) return ans;
    ans = &nodes[*domain_groups[min_domain_size].begin()];
    return ans;

    /* O(n) search for most constrained node */
    // usi min_domain_size = 0;
    // for (int idx = 1; idx <= n; idx++) {
    //   Variable &node = nodes[idx];
    //   if (node.unassigned()) {
    //     usi domain_size = node.domain_stack.top().count();
    //     if (ans == NULL || (min_domain_size > domain_size)) {
    //       ans = &node;
    //       min_domain_size = domain_size;
    //     }
    //   }
    // }
    // return ans;
  }

  bool backtrack_naive(Variable *node, int level = 0) {
    bool found = false;
    if (!node) {
      return level == n || goal();
    }
    for (usi value : values_from_bitmask(node->domain_stack.top())) {
      update_naive(value, node);
      found = backtrack_naive(first_unassigned_variable(), level + 1);
      toc;
      if (found || time_elapsed.count() > 0.5) break;
      update_naive(0, node);
    }
    return found;
  }

  bool backtrack(Variable *node, int level = 0) {
    bool found = false;
    if (!node) {
      return level == n || goal();
    }
    for (usi value : values_from_bitmask(node->domain_stack.top())) {
      update(value, node);                                        // O(deg(node)), optimized for dp to-> O(deg(node) * logN)
      found = backtrack(most_constrained_variable(), level + 1);  // most_cons... O(n), optimized for dp to-> O(1)
      toc;
      if (found || time_elapsed.count() > 0.5) break;
      update(0, node);  // O(deg(node))
    }
    return found;
  }

  vector<int> get_values() {
    vector<int> ans;
    for (int idx = 1; idx <= n; idx++) {
      ans.push_back(nodes[idx].value);
    }
    return ans;
  }

  long double last_epoch = 0;
  bool solve() {
    Variable *start = most_constrained_variable();
    tic;
    bool found = backtrack(start);
    toc;
    last_epoch = time_elapsed.count();

    return found;
  }

  bool solve_naive() {
    Variable *start = first_unassigned_variable();
    tic;
    bool found = backtrack_naive(start);
    toc;
    last_epoch = time_elapsed.count();
    return found;
  }

  bool assigned() {
    bool unassigned_variable = false;
    for (int idx = 1; idx <= n; idx++) {
      unassigned_variable |= nodes[idx].unassigned();
    }
    return !unassigned_variable;
  }

  bool satisfies_constraints() {
    bool violated_constraint = false;
    for (int idx = 1; idx <= n; idx++) {
      Variable *node = &nodes[idx];
      if (!node->unassigned()) {
        usi value = node->value;
        for (auto neighbour : node->contacts) {
          if (!neighbour->unassigned())
            violated_constraint |= neighbour->value == value;
        }
      }
    }
    return !violated_constraint;
  }

  bool goal() {
    return assigned() && satisfies_constraints();
  }
};

vector<int> input_sudoku(string file_name = "") {
  vector<int> v(81, 0);
  if (file_name.size()) {
    ifstream fin(file_name, fstream::in);
    fin >> v;
    fin.close();
  } else {
    cin >> v;
  }
  return v;
}

class GraphSimulatedAnnealing {  // graph with simmulated annealing
 private:
  struct Variable {
    int value, index;
    vector<Variable *> contacts;
    bool locked;
  };
  usi n, mx_value;
  vector<Variable> nodes;

 public:
  GraphSimulatedAnnealing(int n, vector<pair<int, int>> constraints, int mx_value) : n(n), mx_value(mx_value) {
    nodes.resize(n + 1);  // (+ 1) for indexing from 1, ...
    for (auto constraint : constraints) {
      int u = constraint.first, v = constraint.second;
      nodes[u].contacts.push_back(&nodes[v]);
      nodes[v].contacts.push_back(&nodes[u]);
    }
    flush_values();
  };

  void flush_values() {
    int idx = 0;
    for (auto &node : nodes) {
      node.value = 0;
      node.locked = false;
      node.index = idx++;
    }
  }

  void initialise() {
    for (int idx = 1; idx <= n; idx++) {
      auto &node = nodes[idx];
      if (node.locked) continue;
      usi value = 1 + rand() % mx_value;
      node.value = value;
    }
  }

  void initialise(vector<int> values, bool lock = true) {
    for (int idx = 1; idx <= n; idx++) {
      auto &node = nodes[idx];
      usi value = values[idx - 1];
      node.value = value;
      if (lock && value)
        node.locked = true;
    }
  }

  void print() {
    for (int idx = 1; idx <= n; idx++) {
      Variable &node = nodes[idx];
      cout << idx << " " << node.value << "\n";
    }
  }

  vector<int> get_values() {
    vector<int> ans;
    for (int idx = 1; idx <= n; idx++) {
      ans.push_back(nodes[idx].value);
    }
    return ans;
  }

  bool iterate(double beta, int &h_t) {
    int random_index = 1 + rand() % n;
    int random_value = 1 + rand() % mx_value;
    Variable &node = nodes[random_index];
    if (node.locked) return false;
    int initial_value = node.value;

    int prev_conflicts = 0, new_conflicts = 0;
    for (auto neighbour : node.contacts) {
      prev_conflicts += neighbour->value == initial_value;
      new_conflicts += neighbour->value == random_value;
    }

    node.value = random_value;
    int h_new = h_t + new_conflicts - prev_conflicts;

    int delta = h_new - h_t;
    h_t = h_new;
    if (delta <= 0) {
      return true;
    } else {
      double probability = exp(-beta * delta);
      double random_select = (double)rand() / RAND_MAX;
      if (random_select < probability) {
        return true;
      } else {
        node.value = initial_value;
        h_t -= delta;
        return false;
      }
    }
  }

  bool solve() {
    initialise();
    int h_0 = loss_function();
    vector<int> best_assignment = get_values();

    double temperature = 1e1, mx_temperature = 1e1;
    int iterations = 1e4, h = h_0, h_min = h_0, total_rejects = 0;
    for (int idx = 0; idx < iterations; idx++) {
      bool shifted = iterate(1 / temperature, h);
      if (h < h_min) {
        h_min = h;
        best_assignment = get_values();
      }
      total_rejects += shifted ? 1 : -total_rejects;
      temperature -= 1e2 * temperature / (iterations - idx);
      if (total_rejects > 50) {
        temperature = mx_temperature;
        mx_temperature *= 0.9;
        total_rejects = 0;
      }
    }
    initialise(best_assignment, false);
    assert(loss_function() == h_min);
    return h_min == 0;
  }

  bool assigned() {
    bool unassigned_variable = false;
    for (int idx = 1; idx <= n; idx++) {
      unassigned_variable |= nodes[idx].value == 0;
    }
    return !unassigned_variable;
  }

  int loss_function() {
    int violated_constraints = 0;
    for (int idx = 1; idx <= n; idx++) {
      Variable *node = &nodes[idx];
      usi value = node->value;
      for (auto neighbour : node->contacts) {
        violated_constraints += (value && neighbour->value == value);
      }
    }
    return violated_constraints / 2;  // half, because we count one edge twice
  }

  void dump_conflicts() {
    for (int idx = 1; idx <= n; idx++) {
      Variable *node = &nodes[idx];
      usi value = node->value;
      for (auto neighbour : node->contacts) {
        if (value && neighbour->value == value) {
          if (node->locked) {
            neighbour->value = 0;
          } else if (neighbour->locked) {
            node->value = 0;
          } else {
            node->value = 0;
            neighbour->value = 0;
          }
        }
      }
    }
    assert(loss_function() == 0);
  }
};

bool same_locked_values(vector<int> a, vector<int> b) {
  assert(a.size() == b.size());
  int n = a.size(), hamming = 0;
  for (int idx = 0; idx < n; idx++) {
    hamming += (a[idx] && b[idx] && a[idx] != b[idx]);
  }
  return hamming == 0;
}

vector<int> simmulated_annealing(int n, vector<pair<int, int>> constraints, int mx_value, vector<int> initial) {
  int trials = 100;
  for (int trial_idx = 0; trial_idx < trials; trial_idx++) {
    GraphSimulatedAnnealing s(n, constraints, mx_value);
    s.initialise(initial);
    s.solve();
    if (s.loss_function() == 0) return s.get_values();
  }
  return initial;
}

enum Method {
  SIMULATED_ANNEALING,
  BACKTRACK_NAIVE,
  BACKTRACK
};

enum Initial {
  STDIN,
  EMPTY
};

enum Problem {
  CHROMATIC_NUMBER,
  SUDOKU
};

vector<int> chromatic_number(int n, vector<pair<int, int>> constraints, vector<int> initial, int method) {
  int start = 0, end = n;
  vector<int> ans(n);

  auto k_coloring_exists = [&](int k) -> bool {
    bool response = false;
    vector<int> values;
    if (method == Method::SIMULATED_ANNEALING) {
      values = simmulated_annealing(n, constraints, k, initial);
      response = values != initial;
    } else {
      Graph g(n, constraints, k);
      g.initialise(initial);
      if (method == Method::BACKTRACK) {
        response = g.solve();
      } else if (method == Method::BACKTRACK_NAIVE) {
        response = g.solve_naive();
      }
      values = g.get_values();
      assert(same_locked_values(values, initial));
    }
    if (response) ans = values;
    return response;
  };

  assert(k_coloring_exists(n));  // important to initialise ans

  while (start + 1 < end) {
    int k = (start + end) / 2;
    if (k_coloring_exists(k)) {
      end = k;
    } else {
      start = k;
    }
  }

  return ans;
}

void solve_chromatic_number(int method = Method::BACKTRACK, int input = Initial::EMPTY) {
  int n, m;
  cin >> n >> m;
  vector<int> initial(n, 0);
  if (input == Initial::STDIN)
    cin >> initial;
  vector<pair<int, int>> constraints;
  while (m--) {
    int u, v;
    cin >> u >> v;
    constraints.push_back({u, v});
  }

  vector<int> ans;
  int k;

  ans = chromatic_number(n, constraints, initial, method);
  cout << ans;
  k = *max_element(ans.begin(), ans.end());
  cout << k << endl;
}

void solve_sudoku(int method = Method::BACKTRACK) {
  int total_cases;

  cin >> total_cases;
  cout << fixed << setprecision(10);

  int n = 81, k = 9;
  auto sudoku_constraints = sudoku_graph(3);

  long double total_time = 0, success_count = 0;
  for (int index = 1; index <= total_cases; index++) {
    vector<int> initial = input_sudoku(), correct_answer = input_sudoku();
    long double sudoku_time = 0;
    vector<int> solution;

    if (method == Method::SIMULATED_ANNEALING) {
      tic solution = simmulated_annealing(n, sudoku_constraints, k, initial);
      toc sudoku_time = time_elapsed.count();
    } else {
      Graph s(n, sudoku_constraints, k);
      s.initialise(initial);
      if (method == Method::BACKTRACK) {
        s.solve();
      } else if (method == Method::BACKTRACK_NAIVE) {
        s.solve_naive();
      }
      solution = s.get_values();
      sudoku_time = s.last_epoch;
    }

    success_count += solution == correct_answer;
    total_time += sudoku_time;
    cout << "Sudoku " << index << " Passed in " << sudoku_time << "s" << endl;
  }
  cout << "Average Time: " << total_time / total_cases << "s\n";
  cout << "Passed Cases: " << 100 * success_count / total_cases << "%";
}

int main() {
  srand(time(NULL));  // seeding random generator
  // solve_sudoku(Method::SIMULATED_ANNEALING);
  solve_chromatic_number(Method::SIMULATED_ANNEALING, Initial::EMPTY);
  return 0;
}
