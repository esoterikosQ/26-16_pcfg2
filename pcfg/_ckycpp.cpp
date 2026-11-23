// K1 — 확률적 CKY + Viterbi 의 C++(pybind11) 이식. 참조 구현은 pcfg/cky.py.
//
// Jurafsky Fig E.3 의사코드 + Manning 11.3.3 (Viterbi δ/ψ 누적자), 일반화 CKY(단항 클로저).
// Python 엔진과 "같은 트리·비트 단위로 같은 로그확률"을 내기 위한 불변식:
// - 규칙 로그확률은 Python에서 math.log(p)로 미리 계산해 double로 받는다.
// - 셀 = 삽입 순서를 보존하는 label→(logprob, 백포인터) 맵 (Python dict 의미).
//   기존 라벨 갱신은 자리를 유지하고, 새 라벨은 뒤에 붙는다.
// - 순회 순서: width → i → k → 왼쪽 셀 삽입순 B → 오른쪽 셀 삽입순 C → 규칙 목록 순 A.
// - 후보 = (lp_rule + lp_b) + lp_c  (Python `math.log(p) + lp_b + lp_c`의 좌결합 순서).
// - 갱신은 엄격한 `>` (동점이면 먼저 나온 후보 유지).
// - 단항 클로저: 패스 시작 시 셀의 (라벨, 값) 스냅숏을 순회, 변화 없는 패스까지 반복.
// - 컴파일: -ffast-math 금지(덧셈 재결합 금지), -ffp-contract=off (setup.py).
// - 인덱싱: fencepost 0..n, 반열림 구간 [i, j).
//
// 메모리: 셀은 희소(실제로 존재하는 라벨만 담는 벡터 + 라벨 정렬 인덱스). 셀을 채우는 동안만
// 비단말 수 크기의 조밀한 위치 배열(scratch)을 쓴다.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ckycpp {

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// 백포인터 부호화 (Python BackPtr 대응):
//   어휘 ('lex', word)       : k = b = c = -1  (단어는 위치 i의 입력 단어)
//   단항 ('unary', B)        : k = -1, b = B, c = -1
//   이항 ('bin', k, B, C)    : k >= 1, b = B, c = C
struct Entry {
    double lp;
    int32_t label;
    int32_t k;
    int32_t b;
    int32_t c;
};

struct Cell {
    std::vector<Entry> entries;  // 삽입 순서 = Python dict 순서
    std::vector<int32_t> order;  // entries 인덱스를 라벨 오름차순으로 정렬 (조회용)

    int32_t find(int32_t label) const {
        auto it = std::lower_bound(
            order.begin(), order.end(), label,
            [this](int32_t idx, int32_t lab) { return entries[idx].label < lab; });
        if (it != order.end() && entries[*it].label == label) return *it;
        return -1;
    }
};

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

inline void check_csr(const std::vector<int32_t>& off, size_t rows, size_t items, const char* name) {
    if (off.size() != rows + 1)
        throw std::invalid_argument(std::string(name) + ": offsets size mismatch");
    if (off.front() != 0 || static_cast<size_t>(off.back()) != items)
        throw std::invalid_argument(std::string(name) + ": offsets must span [0, items]");
    for (size_t r = 0; r < rows; ++r)
        if (off[r] > off[r + 1])
            throw std::invalid_argument(std::string(name) + ": offsets not monotone");
}

inline void check_ids(const std::vector<int32_t>& v, int32_t lim, const char* name) {
    for (int32_t x : v)
        if (x < 0 || x >= lim)
            throw std::invalid_argument(std::string(name) + ": label id out of range");
}

// 정수 id로 컴파일된 CNF 문법 (RHS 인덱스만; 목록 순서·중복은 Python 인덱스 그대로).
class Grammar {
public:
    Grammar(int32_t n_labels, int32_t n_words,
            std::vector<int32_t> lex_off, std::vector<int32_t> lex_lhs, std::vector<double> lex_lp,
            std::vector<int32_t> un_off, std::vector<int32_t> un_lhs, std::vector<double> un_lp,
            std::vector<int32_t> bin_b, std::vector<int32_t> bin_c, std::vector<int32_t> bin_off,
            std::vector<int32_t> bin_lhs, std::vector<double> bin_lp)
        : n_labels_(n_labels), n_words_(n_words),
          lex_off_(std::move(lex_off)), lex_lhs_(std::move(lex_lhs)), lex_lp_(std::move(lex_lp)),
          un_off_(std::move(un_off)), un_lhs_(std::move(un_lhs)), un_lp_(std::move(un_lp)),
          bin_b_(std::move(bin_b)), bin_c_(std::move(bin_c)), bin_off_(std::move(bin_off)),
          bin_lhs_(std::move(bin_lhs)), bin_lp_(std::move(bin_lp)) {
        if (n_labels_ < 0 || n_words_ < 0)
            throw std::invalid_argument("negative grammar size");
        // 어휘: 단어 id → (A, lp) 목록  (Python grammar._lexical_by_word)
        if (lex_lhs_.size() != lex_lp_.size())
            throw std::invalid_argument("lexical: lhs/lp size mismatch");
        check_csr(lex_off_, static_cast<size_t>(n_words_), lex_lhs_.size(), "lexical");
        check_ids(lex_lhs_, n_labels_, "lexical lhs");
        // 단항: 자식 라벨 id → (A, lp) 목록  (Python grammar._unary_by_child)
        if (un_lhs_.size() != un_lp_.size())
            throw std::invalid_argument("unary: lhs/lp size mismatch");
        check_csr(un_off_, static_cast<size_t>(n_labels_), un_lhs_.size(), "unary");
        check_ids(un_lhs_, n_labels_, "unary lhs");
        // 이항: (B, C) 쌍 p → (A, lp) 목록  (Python grammar._binary_by_rhs)
        if (bin_b_.size() != bin_c_.size())
            throw std::invalid_argument("binary: b/c size mismatch");
        if (bin_lhs_.size() != bin_lp_.size())
            throw std::invalid_argument("binary: lhs/lp size mismatch");
        check_csr(bin_off_, bin_b_.size(), bin_lhs_.size(), "binary");
        check_ids(bin_b_, n_labels_, "binary b");
        check_ids(bin_c_, n_labels_, "binary c");
        check_ids(bin_lhs_, n_labels_, "binary lhs");

        has_unary_.assign(static_cast<size_t>(n_labels_), 0);
        for (int32_t a = 0; a < n_labels_; ++a)
            has_unary_[a] = un_off_[a] != un_off_[a + 1];

        // (B, C) → 쌍 번호: 개방 주소 해시 (선형 탐사)
        has_left_.assign(static_cast<size_t>(n_labels_), 0);
        has_right_.assign(static_cast<size_t>(n_labels_), 0);
        const size_t n_pairs = bin_b_.size();
        size_t cap = 2;
        while (cap < 2 * n_pairs + 2) cap <<= 1;
        hkeys_.assign(cap, EMPTY);
        hvals_.assign(cap, -1);
        hmask_ = cap - 1;
        for (size_t p = 0; p < n_pairs; ++p) {
            const uint64_t key = make_key(bin_b_[p], bin_c_[p]);
            uint64_t h = mix64(key) & hmask_;
            while (hkeys_[h] != EMPTY) {
                if (hkeys_[h] == key)
                    throw std::invalid_argument("binary: duplicate (B, C) pair");
                h = (h + 1) & hmask_;
            }
            hkeys_[h] = key;
            hvals_[h] = static_cast<int32_t>(p);
            if (bin_off_[p] != bin_off_[p + 1]) {
                has_left_[bin_b_[p]] = 1;
                has_right_[bin_c_[p]] = 1;
            }
        }
    }

    int32_t n_labels() const { return n_labels_; }
    int32_t n_words() const { return n_words_; }
    size_t n_binary_pairs() const { return bin_b_.size(); }
    size_t n_binary_rules() const { return bin_lhs_.size(); }
    size_t n_unary_rules() const { return un_lhs_.size(); }
    size_t n_lexical_rules() const { return lex_lhs_.size(); }

    int32_t find_pair(int32_t b, int32_t c) const {
        const uint64_t key = make_key(b, c);
        uint64_t h = mix64(key) & hmask_;
        for (;;) {
            const uint64_t k = hkeys_[h];
            if (k == key) return hvals_[h];
            if (k == EMPTY) return -1;
            h = (h + 1) & hmask_;
        }
    }

    // 파서가 직접 읽는 배열
    int32_t n_labels_, n_words_;
    std::vector<int32_t> lex_off_, lex_lhs_;
    std::vector<double> lex_lp_;
    std::vector<int32_t> un_off_, un_lhs_;
    std::vector<double> un_lp_;
    std::vector<int32_t> bin_b_, bin_c_, bin_off_, bin_lhs_;
    std::vector<double> bin_lp_;
    std::vector<uint8_t> has_unary_, has_left_, has_right_;

private:
    static constexpr uint64_t EMPTY = ~0ULL;  // 라벨 id < 2^31 이므로 실제 키와 겹치지 않음
    static uint64_t make_key(int32_t b, int32_t c) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(c));
    }
    std::vector<uint64_t> hkeys_;
    std::vector<int32_t> hvals_;
    uint64_t hmask_ = 1;
};

class Chart {
public:
    explicit Chart(int32_t n)
        : n_(n), cells_(static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2) {}

    int32_t n() const { return n_; }
    bool valid(int32_t i, int32_t j) const { return 0 <= i && i < j && j <= n_; }

    // 반열림 구간 [i, j) (0 <= i < j <= n) → 삼각 배열 인덱스
    Cell& at(int32_t i, int32_t j) { return cells_[index(i, j)]; }
    const Cell& at(int32_t i, int32_t j) const { return cells_[index(i, j)]; }

    double logprob(int32_t i, int32_t j, int32_t label) const {
        if (!valid(i, j) || label < 0) return NEG_INF;
        const Cell& cell = at(i, j);
        const int32_t e = cell.find(label);
        return e < 0 ? NEG_INF : cell.entries[e].lp;
    }

    // 셀 내용(삽입 순서): (label, logprob, k, b, c)
    std::vector<std::tuple<int32_t, double, int32_t, int32_t, int32_t>> entries(int32_t i,
                                                                               int32_t j) const {
        std::vector<std::tuple<int32_t, double, int32_t, int32_t, int32_t>> out;
        if (!valid(i, j)) return out;
        const Cell& cell = at(i, j);
        out.reserve(cell.entries.size());
        for (const Entry& x : cell.entries) out.emplace_back(x.label, x.lp, x.k, x.b, x.c);
        return out;
    }

    // 백포인터를 따라 [0, n)의 start 트리를 전위 순회로 부호화 (Fig E.3의 BUILD_TREE).
    //   노드 = label, kind (0 어휘 → 뒤에 단어 위치 i / 1 단항 / 2 이항)
    // 파스가 없으면 빈 벡터.
    std::vector<int32_t> tree(int32_t start) const {
        std::vector<int32_t> out;
        if (n_ == 0 || start < 0 || at(0, n_).find(start) < 0) return out;
        size_t total = 0;
        for (const Cell& c : cells_) total += c.entries.size();
        struct Item {
            int32_t i, j, label;
        };
        std::vector<Item> stack;
        stack.push_back(Item{0, n_, start});
        size_t nodes = 0;
        while (!stack.empty()) {
            const Item it = stack.back();
            stack.pop_back();
            if (++nodes > total) throw std::runtime_error("backpointer cycle");
            const Cell& cell = at(it.i, it.j);
            const int32_t e = cell.find(it.label);
            if (e < 0) throw std::runtime_error("dangling backpointer");
            const Entry& x = cell.entries[e];
            out.push_back(it.label);
            if (x.c >= 0) {  // 이항: 왼쪽 서브트리를 먼저 내보내도록 오른쪽을 먼저 쌓는다
                out.push_back(2);
                stack.push_back({x.k, it.j, x.c});
                stack.push_back({it.i, x.k, x.b});
            } else if (x.b >= 0) {  // 단항
                out.push_back(1);
                stack.push_back({it.i, it.j, x.b});
            } else {  // 어휘
                out.push_back(0);
                out.push_back(it.i);
            }
        }
        return out;
    }

private:
    static size_t index(int32_t i, int32_t j) {
        return static_cast<size_t>(j) * static_cast<size_t>(j - 1) / 2 + static_cast<size_t>(i);
    }
    int32_t n_;
    std::vector<Cell> cells_;
};

class Parser {
public:
    Parser(const Grammar& g, Chart& chart)
        : g_(g), chart_(chart), pos_(static_cast<size_t>(g.n_labels()), -1) {}

    void run(const std::vector<int32_t>& words) {
        const int32_t n = chart_.n();

        // --- 어휘 규칙: 폭 1 셀 [i, i+1) 채우기 (Fig E.3의 첫 루프) ---
        for (int32_t i = 0; i < n; ++i) {
            Cell& cell = chart_.at(i, i + 1);
            const int32_t w = words[i];
            if (w >= 0) {
                for (int32_t r = g_.lex_off_[w]; r < g_.lex_off_[w + 1]; ++r)
                    relax(cell.entries, g_.lex_lhs_[r], g_.lex_lp_[r], -1, -1, -1);
            }
            unary_closure(cell.entries);
            finalize(cell);
        }

        // --- 이항 규칙: 폭 2..n 셀 채우기 (Fig E.3의 삼중 루프) ---
        for (int32_t width = 2; width <= n; ++width) {
            for (int32_t i = 0; i + width <= n; ++i) {
                const int32_t j = i + width;
                Cell& cell = chart_.at(i, j);
                for (int32_t k = i + 1; k < j; ++k) {
                    const std::vector<Entry>& left = chart_.at(i, k).entries;
                    const std::vector<Entry>& right = chart_.at(k, j).entries;
                    if (left.empty() || right.empty()) continue;
                    // 오른쪽 자식이 될 수 없는 라벨은 갱신을 일으키지 않으므로 순서를 지켜 걸러 둔다.
                    rfilt_.clear();
                    for (const Entry& y : right)
                        if (g_.has_right_[y.label]) rfilt_.emplace_back(y.label, y.lp);
                    if (rfilt_.empty()) continue;
                    for (const Entry& x : left) {
                        if (!g_.has_left_[x.label]) continue;
                        const int32_t B = x.label;
                        const double lp_b = x.lp;
                        for (const auto& rc : rfilt_) {
                            const int32_t p = g_.find_pair(B, rc.first);
                            if (p < 0) continue;
                            const double lp_c = rc.second;
                            for (int32_t r = g_.bin_off_[p]; r < g_.bin_off_[p + 1]; ++r) {
                                const double cand = (g_.bin_lp_[r] + lp_b) + lp_c;
                                relax(cell.entries, g_.bin_lhs_[r], cand, k, B, rc.first);
                            }
                        }
                    }
                }
                // 단항 규칙 처리 (일반화 CKY): 이항 조합 뒤 단항 클로저 적용.
                unary_closure(cell.entries);
                finalize(cell);
            }
        }
    }

private:
    // Python: if cand > cell.get(A, NEG_INF): cell[A] = cand; back[A] = ...
    bool relax(std::vector<Entry>& e, int32_t a, double cand, int32_t k, int32_t b, int32_t c) {
        const int32_t p = pos_[a];
        const double cur = p < 0 ? NEG_INF : e[p].lp;
        if (!(cand > cur)) return false;
        if (p < 0) {
            pos_[a] = static_cast<int32_t>(e.size());
            e.push_back(Entry{cand, a, k, b, c});
        } else {
            e[p] = Entry{cand, a, k, b, c};
        }
        return true;
    }

    // 단항 클로저 (pcfg/cky.py `_apply_unary`): `for B, lp_b in list(cell.items())` 스냅숏.
    void unary_closure(std::vector<Entry>& e) {
        bool changed = true;
        while (changed) {
            changed = false;
            snap_.clear();
            for (const Entry& x : e)
                if (g_.has_unary_[x.label]) snap_.emplace_back(x.label, x.lp);
            for (const auto& s : snap_) {
                const int32_t B = s.first;
                for (int32_t r = g_.un_off_[B]; r < g_.un_off_[B + 1]; ++r) {
                    const double cand = g_.un_lp_[r] + s.second;
                    if (relax(e, g_.un_lhs_[r], cand, -1, B, -1)) changed = true;
                }
            }
        }
    }

    void finalize(Cell& cell) {
        for (const Entry& x : cell.entries) pos_[x.label] = -1;
        cell.entries.shrink_to_fit();
        cell.order.resize(cell.entries.size());
        std::iota(cell.order.begin(), cell.order.end(), 0);
        const std::vector<Entry>& e = cell.entries;
        std::sort(cell.order.begin(), cell.order.end(),
                  [&e](int32_t a, int32_t b) { return e[a].label < e[b].label; });
    }

    const Grammar& g_;
    Chart& chart_;
    std::vector<int32_t> pos_;  // scratch: 라벨 → 채우는 중인 셀의 entries 인덱스 (-1 = 없음)
    std::vector<std::pair<int32_t, double>> snap_;
    std::vector<std::pair<int32_t, double>> rfilt_;
};

inline std::unique_ptr<Chart> parse_ids(const Grammar& g, const std::vector<int32_t>& words) {
    for (int32_t w : words)
        if (w < -1 || w >= g.n_words()) throw std::invalid_argument("word id out of range");
    auto chart = std::make_unique<Chart>(static_cast<int32_t>(words.size()));
    Parser(g, *chart).run(words);
    return chart;
}

}  // namespace ckycpp

PYBIND11_MODULE(_ckycpp, m) {
    using namespace ckycpp;
    m.doc() = "K1 C++ CKY/Viterbi engine (reference: pcfg/cky.py). Use via pcfg.cky_cpp.";

    py::class_<Grammar>(m, "Grammar")
        .def(py::init<int32_t, int32_t,
                      std::vector<int32_t>, std::vector<int32_t>, std::vector<double>,
                      std::vector<int32_t>, std::vector<int32_t>, std::vector<double>,
                      std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>,
                      std::vector<int32_t>, std::vector<double>>(),
             py::arg("n_labels"), py::arg("n_words"),
             py::arg("lex_off"), py::arg("lex_lhs"), py::arg("lex_lp"),
             py::arg("un_off"), py::arg("un_lhs"), py::arg("un_lp"),
             py::arg("bin_b"), py::arg("bin_c"), py::arg("bin_off"),
             py::arg("bin_lhs"), py::arg("bin_lp"))
        .def_property_readonly("n_labels", &Grammar::n_labels)
        .def_property_readonly("n_words", &Grammar::n_words)
        .def_property_readonly("n_binary_pairs", &Grammar::n_binary_pairs)
        .def_property_readonly("n_binary_rules", &Grammar::n_binary_rules)
        .def_property_readonly("n_unary_rules", &Grammar::n_unary_rules)
        .def_property_readonly("n_lexical_rules", &Grammar::n_lexical_rules);

    py::class_<Chart>(m, "Chart")
        .def_property_readonly("n", &Chart::n)
        .def("logprob", &Chart::logprob, py::arg("i"), py::arg("j"), py::arg("label"))
        .def("entries", &Chart::entries, py::arg("i"), py::arg("j"))
        .def("tree", &Chart::tree, py::arg("start"));

    m.def(
        "parse",
        [](const Grammar& g, const std::vector<int32_t>& words) {
            py::gil_scoped_release nogil;
            return parse_ids(g, words);
        },
        py::arg("grammar"), py::arg("word_ids"),
        "Parse word ids (-1 = not in lexicon). Returns the filled Chart.");
}
