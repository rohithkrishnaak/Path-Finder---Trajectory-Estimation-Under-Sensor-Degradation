// Streaming input: zip (via miniz, never fully extracted), folder (searched for imu0/data.csv) or CSV.
// Rows are parsed with a tolerant parser that mirrors research/common.py::parse_imu_text.
#pragma once

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>

#include "miniz.h"

namespace pf {

struct Row {
    int64_t t = 0;
    double g[3] = {0, 0, 0};
    double a[3] = {0, 0, 0};
};

static inline bool ends_with_ci(const std::string& s, const char* suf) {
    size_t n = std::strlen(suf);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        char c = s[s.size() - n + i];
        if (c == '\\') c = '/';
        if (std::tolower((unsigned char)c) != std::tolower((unsigned char)suf[i])) return false;
    }
    return true;
}

// Byte source with a fixed 64 KiB buffer.
class ByteSource {
public:
    ~ByteSource() { close(); }

    bool open(const std::string& path, std::string& err) {
        close();
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::is_directory(path, ec)) {
            for (auto it = fs::recursive_directory_iterator(path, ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
                if (ec) break;
                const auto& p = it->path();
                if (p.filename() == "data.csv" && p.parent_path().filename() == "imu0") return open_file(p.string(), err);
            }
            err = "no imu0/data.csv under " + path;
            return false;
        }
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open " + path; return false; }
        unsigned char sig[4] = {0, 0, 0, 0};
        size_t got = std::fread(sig, 1, 4, f);
        std::fclose(f);
        if (got == 4 && sig[0] == 'P' && sig[1] == 'K' && sig[2] == 3 && sig[3] == 4) return open_zip(path, err);
        return open_file(path, err);
    }

    // Returns false at EOF. Strips trailing CR/LF.
    bool getline(std::string& line) {
        line.clear();
        for (;;) {
            if (pos_ >= len_) {
                if (!refill()) return !line.empty();
            }
            const char* start = buf_ + pos_;
            const char* nl = static_cast<const char*>(std::memchr(start, '\n', len_ - pos_));
            if (nl) {
                line.append(start, nl - start);
                pos_ += (nl - start) + 1;
                while (!line.empty() && line.back() == '\r') line.pop_back();
                return true;
            }
            line.append(start, len_ - pos_);
            pos_ = len_;
        }
    }

    void close() {
        if (iter_) { mz_zip_reader_extract_iter_free(iter_); iter_ = nullptr; }
        if (zip_open_) { mz_zip_reader_end(&zip_); zip_open_ = false; }
        if (file_) { std::fclose(file_); file_ = nullptr; }
        pos_ = len_ = 0;
    }

private:
    bool open_file(const std::string& p, std::string& err) {
        file_ = std::fopen(p.c_str(), "rb");
        if (!file_) { err = "cannot open " + p; return false; }
        return true;
    }

    bool open_zip(const std::string& p, std::string& err) {
        std::memset(&zip_, 0, sizeof(zip_));
        if (!mz_zip_reader_init_file(&zip_, p.c_str(), 0)) { err = "bad zip " + p; return false; }
        zip_open_ = true;
        mz_uint n = mz_zip_reader_get_num_files(&zip_);
        int best = -1, best_csv = -1;
        size_t best_len = 0, best_csv_len = 0;
        char name[1024];
        for (mz_uint i = 0; i < n; ++i) {
            if (mz_zip_reader_is_file_a_directory(&zip_, i)) continue;
            mz_zip_reader_get_filename(&zip_, i, name, sizeof(name));
            std::string s(name);
            if (ends_with_ci(s, "imu0/data.csv") && (best < 0 || s.size() < best_len)) { best = (int)i; best_len = s.size(); }
            if (ends_with_ci(s, ".csv") && (best_csv < 0 || s.size() < best_csv_len)) { best_csv = (int)i; best_csv_len = s.size(); }
        }
        if (best < 0) best = best_csv;
        if (best < 0) { err = "no imu0/data.csv in zip"; return false; }
        iter_ = mz_zip_reader_extract_iter_new(&zip_, (mz_uint)best, 0);
        if (!iter_) { err = "cannot stream zip entry"; return false; }
        return true;
    }

    bool refill() {
        pos_ = 0;
        if (file_) len_ = std::fread(buf_, 1, sizeof(buf_), file_);
        else if (iter_) len_ = mz_zip_reader_extract_iter_read(iter_, buf_, sizeof(buf_));
        else len_ = 0;
        return len_ > 0;
    }

    FILE* file_ = nullptr;
    mz_zip_archive zip_{};
    bool zip_open_ = false;
    mz_zip_reader_extract_iter_state* iter_ = nullptr;
    char buf_[1 << 16];
    size_t pos_ = 0, len_ = 0;
};

static inline std::string strip(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

static inline double to_float(const std::string& field) {
    std::string s = strip(field);
    if (s.empty()) return std::numeric_limits<double>::quiet_NaN();
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return std::numeric_limits<double>::quiet_NaN();
    return v;
}

// Parse one line. Returns false for blank/comment/unparseable (header) lines.
static inline bool parse_row(const std::string& raw, Row& r) {
    std::string line = strip(raw);
    if (line.empty() || line[0] == '#') return false;
    std::string norm = line;
    for (char& c : norm) if (c == ';' || c == '\t') c = ',';
    std::string parts[8];
    int np = 0;
    if (norm.find(',') != std::string::npos) {
        size_t start = 0;
        while (np < 8) {
            size_t k = norm.find(',', start);
            parts[np++] = norm.substr(start, k == std::string::npos ? std::string::npos : k - start);
            if (k == std::string::npos) break;
            start = k + 1;
        }
    } else {
        size_t i = 0;
        while (i < line.size() && np < 8) {
            while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
            size_t j = i;
            while (j < line.size() && !std::isspace((unsigned char)line[j])) ++j;
            if (j > i) parts[np++] = line.substr(i, j - i);
            i = j;
        }
    }
    std::string t0 = strip(parts[0]);
    if (t0.empty()) return false;
    char* end = nullptr;
    errno = 0;
    long long t = std::strtoll(t0.c_str(), &end, 10);
    if (end == t0.c_str() || *end != '\0' || errno == ERANGE) {
        double d = to_float(t0);
        if (!std::isfinite(d)) return false;
        t = std::llround(d);
    }
    r.t = t;
    double v[6];
    for (int k = 0; k < 6; ++k) v[k] = (k + 1 < np) ? to_float(parts[k + 1]) : std::numeric_limits<double>::quiet_NaN();
    for (int k = 0; k < 3; ++k) { r.g[k] = v[k]; r.a[k] = v[k + 3]; }
    return true;
}

// Row stream: wraps a ByteSource and yields parsed rows.
class RowReader {
public:
    bool open(const std::string& path, std::string& err) { return src_.open(path, err); }
    bool next(Row& r) {
        while (src_.getline(line_)) {
            if (parse_row(line_, r)) return true;
        }
        return false;
    }
private:
    ByteSource src_;
    std::string line_;
};

}  // namespace pf
