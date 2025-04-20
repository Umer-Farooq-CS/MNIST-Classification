#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VERSIONS 5
#define MAX_LINE_LENGTH 256

// Structure to store version data
typedef struct {
    int version;
    double runtime;
    double speedup;
} VersionData;

// Function to extract runtime from a file
double extract_runtime(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return -1.0;
    }
    
    char line[MAX_LINE_LENGTH];
    double runtime = -1.0;
    
    // Read the single line in the file
    if (fgets(line, MAX_LINE_LENGTH, file)) {
        // Extract the time value
        char* time_str = strstr(line, "time:");
        if (time_str) {
            time_str += 5; // Move past "time:"
            char* seconds_str = strstr(time_str, "s");
            if (seconds_str) {
                *seconds_str = '\0'; // Replace 's' with null terminator
                runtime = atof(time_str);
            }
        }
    }
    
    fclose(file);
    return runtime;
}

// Function to compare version data for sorting
int compare_versions(const void* a, const void* b) {
    return ((VersionData*)a)->version - ((VersionData*)b)->version;
}

int main() {
    VersionData versions[MAX_VERSIONS];
    int valid_versions = 0;
    double baseline_runtime = -1.0;
    
    printf("Reading runtime data for versions V1 to V5...\n\n");
    
    // Collect runtime data for each version
    for (int i = 1; i <= MAX_VERSIONS; i++) {
        char filename[256];
        sprintf(filename, "../V%d/build/training_time.txt", i);
        
        double runtime = extract_runtime(filename);
        if (runtime > 0) {
            versions[valid_versions].version = i;
            versions[valid_versions].runtime = runtime;
            if (i == 1) {
                baseline_runtime = runtime;
            }
            valid_versions++;
        } else {
            printf("Warning: Could not extract runtime from V%d/build/training_time.txt\n", i);
        }
    }
    
    // Check if we found the baseline runtime
    if (baseline_runtime <= 0) {
        printf("Error: Could not determine baseline runtime from V1.\n");
        return 1;
    }
    
    // Calculate speedup for each version
    for (int i = 0; i < valid_versions; i++) {
        versions[i].speedup = baseline_runtime / versions[i].runtime;
    }
    
    // Sort by version number
    qsort(versions, valid_versions, sizeof(VersionData), compare_versions);
    
    // Print the results in a tabular format
    printf("╔═════════════════════════════════════════════════════╗\n");
    printf("║                 SPEEDUP ANALYSIS                    ║\n");
    printf("╠═══════════╦═══════════════════╦═══════════════════╣\n");
    printf("║  Version  ║   Runtime (sec)   ║      Speedup      ║\n");
    printf("╠═══════════╬═══════════════════╬═══════════════════╣\n");
    
    for (int i = 0; i < valid_versions; i++) {
        printf("║    V%-2d    ║     %9.3f     ║", versions[i].version, versions[i].runtime);
        
        if (versions[i].version == 1) {
            printf("     baseline     ║\n");
        } else {
            printf("     %6.2fx      ║\n", versions[i].speedup);
        }
    }
    
    printf("╚═══════════╩═══════════════════╩═══════════════════╝\n");
    
    // Find the version with the best speedup
    int best_version = 1;
    double best_speedup = 1.0;
    
    for (int i = 0; i < valid_versions; i++) {
        if (versions[i].speedup > best_speedup) {
            best_speedup = versions[i].speedup;
            best_version = versions[i].version;
        }
    }
    
    printf("\nSummary:\n");
    printf("- Baseline (V1) runtime: %.3f seconds\n", baseline_runtime);
    
    if (best_version > 1) {
        printf("- Best performance: V%d with %.2fx speedup\n", best_version, best_speedup);
    } else {
        printf("- No version outperformed the baseline\n");
    }
    
    return 0;
}