#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DIMS 14

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
        p++;
    }
    return p;
}

static const char *find_value(const char *start, const char *key) {
    const char *p = strstr(start, key);
    if (p == NULL) {
        return NULL;
    }
    p += strlen(key);
    p = skip_ws(p);
    if (*p != ':') {
        return NULL;
    }
    return skip_ws(p + 1);
}

static int parse_number_key(const char *start, const char *key, double *out) {
    const char *p = find_value(start, key);
    char *endptr = NULL;
    if (p == NULL) {
        return 0;
    }
    *out = strtod(p, &endptr);
    return endptr != p;
}

static int parse_long_key(const char *start, const char *key, long *out) {
    const char *p = find_value(start, key);
    char *endptr = NULL;
    if (p == NULL) {
        return 0;
    }
    *out = strtol(p, &endptr, 10);
    return endptr != p;
}

static int parse_bool_key(const char *start, const char *key, int *out) {
    const char *p = find_value(start, key);
    if (p == NULL) {
        return 0;
    }
    if (strncmp(p, "true", 4) == 0) {
        *out = 1;
        return 1;
    }
    if (strncmp(p, "false", 5) == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int parse_string_key(const char *start, const char *key, const char **out, Py_ssize_t *out_len) {
    const char *p = find_value(start, key);
    const char *q = NULL;
    if (p == NULL || *p != '"') {
        return 0;
    }
    p++;
    q = strchr(p, '"');
    if (q == NULL) {
        return 0;
    }
    *out = p;
    *out_len = q - p;
    return 1;
}

static int parse_array_bounds(const char *start, const char *key, const char **out, const char **out_end) {
    const char *p = find_value(start, key);
    const char *q = NULL;
    if (p == NULL || *p != '[') {
        return 0;
    }
    p++;
    q = strchr(p, ']');
    if (q == NULL) {
        return 0;
    }
    *out = p;
    *out_end = q;
    return 1;
}

static int array_has_string(const char *p, const char *end, const char *needle, Py_ssize_t needle_len) {
    while (p < end) {
        if (*p == '"') {
            const char *s = p + 1;
            const char *q = memchr(s, '"', (size_t)(end - s));
            if (q == NULL) {
                return 0;
            }
            if ((Py_ssize_t)(q - s) == needle_len && memcmp(s, needle, (size_t)needle_len) == 0) {
                return 1;
            }
            p = q + 1;
        } else {
            p++;
        }
    }
    return 0;
}

static int parse2(const char *p) {
    return (p[0] - '0') * 10 + (p[1] - '0');
}

static int parse4(const char *p) {
    return (p[0] - '0') * 1000 + (p[1] - '0') * 100 + (p[2] - '0') * 10 + (p[3] - '0');
}

static long long days_from_civil(int year, int month, int day) {
    int era;
    unsigned yoe;
    unsigned mp;
    unsigned doy;
    unsigned doe;

    year -= month <= 2;
    era = (year >= 0 ? year : year - 399) / 400;
    yoe = (unsigned)(year - era * 400);
    mp = (unsigned)(month + (month > 2 ? -3 : 9));
    doy = (153 * mp + 2) / 5 + (unsigned)day - 1;
    doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return (long long)era * 146097 + (long long)doe - 719468;
}

static long long epoch_seconds(const char *ts) {
    int year = parse4(ts);
    int month = parse2(ts + 5);
    int day = parse2(ts + 8);
    int hour = parse2(ts + 11);
    int minute = parse2(ts + 14);
    int second = parse2(ts + 17);
    return (((days_from_civil(year, month, day) * 24 + hour) * 60 + minute) * 60 + second);
}

static int day_of_week_monday_zero(int year, int month, int day) {
    static const int table[12] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int y = year - (month < 3 ? 1 : 0);
    int dow = (y + y / 4 - y / 100 + y / 400 + table[month - 1] + day) % 7;
    return (dow + 6) % 7;
}

static double clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static double round4(double value) {
    return floor(value * 10000.0 + 0.5) / 10000.0;
}

static double mcc_risk(const char *mcc, Py_ssize_t len) {
    if (len != 4) {
        return 0.50;
    }
    if (memcmp(mcc, "5411", 4) == 0) return 0.15;
    if (memcmp(mcc, "5812", 4) == 0) return 0.30;
    if (memcmp(mcc, "5912", 4) == 0) return 0.20;
    if (memcmp(mcc, "5944", 4) == 0) return 0.45;
    if (memcmp(mcc, "7801", 4) == 0) return 0.80;
    if (memcmp(mcc, "7802", 4) == 0) return 0.75;
    if (memcmp(mcc, "7995", 4) == 0) return 0.85;
    if (memcmp(mcc, "4511", 4) == 0) return 0.35;
    if (memcmp(mcc, "5311", 4) == 0) return 0.25;
    if (memcmp(mcc, "5999", 4) == 0) return 0.50;
    return 0.50;
}

static int build_vector(const char *json, double *vector) {
    const char *transaction = find_value(json, "\"transaction\"");
    const char *customer = find_value(json, "\"customer\"");
    const char *merchant = find_value(json, "\"merchant\"");
    const char *terminal = find_value(json, "\"terminal\"");
    const char *last_transaction = find_value(json, "\"last_transaction\"");
    const char *requested_at = NULL;
    const char *last_timestamp = NULL;
    const char *merchant_id = NULL;
    const char *mcc = NULL;
    const char *known_start = NULL;
    const char *known_end = NULL;
    Py_ssize_t requested_at_len = 0;
    Py_ssize_t last_timestamp_len = 0;
    Py_ssize_t merchant_id_len = 0;
    Py_ssize_t mcc_len = 0;
    double amount = 0.0;
    double customer_avg_amount = 0.0;
    double merchant_avg_amount = 0.0;
    double km_from_home = 0.0;
    double km_from_current = 0.0;
    long installments = 0;
    long tx_count_24h = 0;
    int is_online = 0;
    int card_present = 0;
    int year;
    int month;
    int day;
    int hour;
    int i;

    if (transaction == NULL || customer == NULL || merchant == NULL || terminal == NULL || last_transaction == NULL) {
        return 0;
    }
    if (!parse_number_key(transaction, "\"amount\"", &amount)) return 0;
    if (!parse_long_key(transaction, "\"installments\"", &installments)) return 0;
    if (!parse_string_key(transaction, "\"requested_at\"", &requested_at, &requested_at_len)) return 0;
    if (requested_at_len < 19) return 0;

    if (!parse_number_key(customer, "\"avg_amount\"", &customer_avg_amount)) return 0;
    if (!parse_long_key(customer, "\"tx_count_24h\"", &tx_count_24h)) return 0;
    if (!parse_array_bounds(customer, "\"known_merchants\"", &known_start, &known_end)) return 0;

    if (!parse_string_key(merchant, "\"id\"", &merchant_id, &merchant_id_len)) return 0;
    if (!parse_string_key(merchant, "\"mcc\"", &mcc, &mcc_len)) return 0;
    if (!parse_number_key(merchant, "\"avg_amount\"", &merchant_avg_amount)) return 0;

    if (!parse_bool_key(terminal, "\"is_online\"", &is_online)) return 0;
    if (!parse_bool_key(terminal, "\"card_present\"", &card_present)) return 0;
    if (!parse_number_key(terminal, "\"km_from_home\"", &km_from_home)) return 0;

    year = parse4(requested_at);
    month = parse2(requested_at + 5);
    day = parse2(requested_at + 8);
    hour = parse2(requested_at + 11);

    vector[0] = clamp01(amount / 10000.0);
    vector[1] = clamp01((double)installments / 12.0);
    vector[2] = customer_avg_amount > 0.0 ? clamp01((amount / customer_avg_amount) / 10.0) : 1.0;
    vector[3] = (double)hour / 23.0;
    vector[4] = (double)day_of_week_monday_zero(year, month, day) / 6.0;

    if (strncmp(last_transaction, "null", 4) == 0) {
        vector[5] = -1.0;
        vector[6] = -1.0;
    } else {
        double minutes;
        if (!parse_string_key(last_transaction, "\"timestamp\"", &last_timestamp, &last_timestamp_len)) return 0;
        if (last_timestamp_len < 19) return 0;
        if (!parse_number_key(last_transaction, "\"km_from_current\"", &km_from_current)) return 0;
        minutes = (double)(epoch_seconds(requested_at) - epoch_seconds(last_timestamp)) / 60.0;
        vector[5] = clamp01(minutes / 1440.0);
        vector[6] = clamp01(km_from_current / 1000.0);
    }

    vector[7] = clamp01(km_from_home / 1000.0);
    vector[8] = clamp01((double)tx_count_24h / 20.0);
    vector[9] = is_online ? 1.0 : 0.0;
    vector[10] = card_present ? 1.0 : 0.0;
    vector[11] = array_has_string(known_start, known_end, merchant_id, merchant_id_len) ? 0.0 : 1.0;
    vector[12] = mcc_risk(mcc, mcc_len);
    vector[13] = clamp01(merchant_avg_amount / 10000.0);

    for (i = 0; i < DIMS; i++) {
        vector[i] = round4(vector[i]);
    }
    return 1;
}

static PyObject *rinha_parse(PyObject *self, PyObject *args) {
    const char *json = NULL;
    Py_ssize_t json_len = 0;
    double vector[DIMS];

    (void)self;
    if (!PyArg_ParseTuple(args, "y#", &json, &json_len)) {
        return NULL;
    }
    (void)json_len;

    if (!build_vector(json, vector)) {
        Py_RETURN_NONE;
    }
    return PyBytes_FromStringAndSize((const char *)vector, sizeof(vector));
}

static PyMethodDef RinhaMethods[] = {
    {"parse", rinha_parse, METH_VARARGS, "Parse a fraud-score JSON payload into the normalized vector."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef RinhaModule = {
    PyModuleDef_HEAD_INIT,
    "rinha_native",
    NULL,
    -1,
    RinhaMethods,
};

PyMODINIT_FUNC PyInit_rinha_native(void) {
    return PyModule_Create(&RinhaModule);
}
