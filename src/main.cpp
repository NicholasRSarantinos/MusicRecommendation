#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include <string>
#include "tnt/tnt2vector.h"
#include <time.h>
#include <chrono>
#include <dirent.h>
#include <stdio.h>
#include <sqlite3.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits>
#include <fcntl.h>
#include <float.h>

using namespace std;
using namespace TNT;
using namespace essentia;
using namespace essentia::standard;

#ifdef _WIN32
    #define seperator "\\"
#else
    #define seperator "/"
#endif

sqlite3 *db;

void close_db()
{
    sqlite3_close(db);
}

void load_db()
{
    char *zErrMsg = 0;
    
    if(sqlite3_open("music.db", &db))
    {
        cout << "Can't open database\n";
        exit(1);
    }
    else if(sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS music (path TEXT PRIMARY KEY, data BLOB NOT NULL);", NULL, 0, &zErrMsg) != SQLITE_OK)
    {
        cout << "SQL ERROR: " << zErrMsg << "\n";
        sqlite3_free(zErrMsg);
        exit(1);
    }
}

void store_song_data(string &song_path, float* song_data, size_t song_items)
{
    sqlite3_stmt *stmt = NULL;
    
    if (sqlite3_prepare(db, "INSERT INTO music(path, data) VALUES(?, ?)", -1, &stmt, NULL) != SQLITE_OK ||
        sqlite3_bind_text(stmt, 1, song_path.c_str(), song_path.size(), SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_bind_blob(stmt, 2, (char*)song_data, song_items*sizeof(float), SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_step(stmt) != SQLITE_DONE
       )
    {
        cerr << "SQL ERROR WHEN STORING: " << sqlite3_errmsg(db) << endl;
        exit(1);
    }
    else
        sqlite3_finalize(stmt);
}

bool read_song_data(string &song_path, float* song_data, size_t song_items)//true if we got songs, false otherwise
{
    sqlite3_stmt *stmt = NULL;
    
    if (sqlite3_prepare(db, "SELECT data FROM music WHERE path = ? LIMIT 1", -1, &stmt, NULL) || //limit 1 is used for performance reasons
        sqlite3_bind_text(stmt, 1, song_path.c_str(), -1, SQLITE_STATIC) != SQLITE_OK)
    {
        cerr << "SQL ERROR WHEN READING: " << sqlite3_errmsg(db) << endl;
        exit(1);
    }
    else if(sqlite3_step(stmt) != SQLITE_ROW)
    {
        sqlite3_finalize(stmt);
        return false;
    }
    else
    {
        float* mem_got = (float*)sqlite3_column_blob(stmt, 0);
        memcpy((char*)song_data, (char*)mem_got, song_items*sizeof(float));
        sqlite3_finalize(stmt);
        return true;
    }
}

int count_stored_songs()
{
    sqlite3_stmt *stmt = NULL;
    
    if (sqlite3_prepare(db, "SELECT count(1) FROM music", -1, &stmt, NULL) ||
       sqlite3_step(stmt) != SQLITE_ROW)
    {
        cerr << "SQL ERROR WHEN READING: " << sqlite3_errmsg(db) << endl;
        exit(1);
    }
    else
    {
        int res = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
        return res;
    }
}

void print(vector<Real> &vector, string &name)
{
    cout << "\nPrinting Vector '"+name+"'\n\n";
    
    for(size_t i = 0; i < vector.size(); i++)
        cout << vector[i] << " ";
    
    cout << "\n\n";
}

void print(vector< vector<Real> > &matrix, string &name)
{
    cout << "\nPrinting Matrix '"+name+"'\n\n";
    
    for(size_t i = 0; i < matrix.size(); i++)
    {
        for(size_t j = 0; j < matrix[i].size(); j++)
            cout << matrix[i][j] << " ";
        
        cout << "\n";
    }
    
    cout << "\n\n";
}

vector< vector<Real> > sum_or_sub_matrixes(bool is_sum_else_sub, vector< vector<Real> > &matrix_X, vector< vector<Real> > &matrix_Y)
{
    vector< vector<Real> > matrix_OUT(matrix_X.size(), std::vector<Real>(matrix_X[0]));
    
    for(size_t i = 0; i < matrix_X.size(); i++)
    {
        vector<Real> temp_arr(matrix_X[i].size());

        for(size_t j = 0; j < matrix_X[i].size(); j++)
            temp_arr[j] = is_sum_else_sub ? (matrix_X[i][j]+matrix_Y[i][j]) : (matrix_X[i][j]-matrix_Y[i][j]);

        matrix_OUT[i] = temp_arr;
    }
    
    return matrix_OUT;
}

vector< vector<Real> > transpose_matrix(vector< vector<Real> > &matrix)
{
    vector< vector<Real> > matrix_transposed(matrix[0].size(), std::vector<Real>(matrix.size()));
    
    for(size_t i = 0; i < matrix.size(); i++)
        for(size_t j = 0; j < matrix[i].size(); j++)
            matrix_transposed[j][i] = matrix[i][j];
    
    return matrix_transposed;
}

vector< vector<Real> > array_to_matrix(vector<Real> &input_array)
{
    vector< vector<Real> > matrix(1, std::vector<Real>(input_array.size()));
    
    for(size_t i = 0; i < input_array.size(); i++)
        matrix[0][i] = input_array[i];
    
    return matrix;
}

vector< vector<Real> > multiply_matrixes(vector< vector<Real> > &matrix_X, vector< vector<Real> > &matrix_Y)
{
    size_t 
        matrix_X_rows = matrix_X.size(),
        matrix_X_columns = matrix_X[0].size(),
        matrix_Y_columns = matrix_Y[0].size();
    
    vector< vector<Real> > output_matrix(matrix_X_rows, std::vector<Real>(matrix_Y_columns));
    
    for(size_t i = 0; i < matrix_X_rows; i++)
        for(size_t j = 0; j < matrix_Y_columns; j++)
        {
            output_matrix[i][j] = 0;
            
            for(size_t k = 0; k < matrix_X_columns; k++)
                output_matrix[i][j] += matrix_X[i][k] * matrix_Y[k][j];
        }
    
    return output_matrix;
}

vector< vector<Real> > multiply_matrix_by_scalar(vector< vector<Real> > &matrix, Real &scalar)
{
    vector< vector<Real> > output_matrix(matrix.size(), std::vector<Real>(matrix[0].size()));
    
    for(size_t i = 0; i < output_matrix.size(); i++)
        for(size_t j = 0; j < output_matrix[i].size(); j++)
            output_matrix[i][j] = matrix[i][j]*scalar;
    
    return output_matrix;
}

Real trace_matrix(vector< vector<Real> > &matrix)
{
    Real sum = 0;
    
    for(size_t i = 0; i < matrix.size(); i++)
        sum += matrix[i][i];
    
    return sum;
}

float slow_get_song_distance(
    vector< vector<Real> > &SONG_X_MFCC_Covariance_Matrix, vector< vector<Real> > &SONG_X_Inverse_MFCC_Covariance_Matrix, vector<Real> &SONG_X_MFCC_Means_array,
    vector< vector<Real> > &SONG_Y_MFCC_Covariance_Matrix, vector< vector<Real> > &SONG_Y_Inverse_MFCC_Covariance_Matrix, vector<Real> &SONG_Y_MFCC_Means_array
)
{
    //the functions do no error checking, be carefull how you use them
    //we are implementing what is written at line http://mtg.upf.edu/system/files/publications/dbogdanov_phd_thesis_2013_0.pdf AT LINE 81
    
    vector <vector<Real> > 
        SONG_X_MFCC_Means = array_to_matrix(SONG_X_MFCC_Means_array),
        SONG_Y_MFCC_Means = array_to_matrix(SONG_Y_MFCC_Means_array),
        
        X_MINUS_Y_MFCC_Means = sum_or_sub_matrixes(false, SONG_X_MFCC_Means, SONG_Y_MFCC_Means),
        
        X_SUM_Y_Inverse_MFCC_Covariance_matrix = sum_or_sub_matrixes(true, SONG_X_Inverse_MFCC_Covariance_Matrix, SONG_Y_Inverse_MFCC_Covariance_Matrix),
        
        transposed_X_MINUS_Y_MFCC_Means = transpose_matrix(X_MINUS_Y_MFCC_Means),
        
        to_trace_one = multiply_matrixes(SONG_X_Inverse_MFCC_Covariance_Matrix, SONG_Y_MFCC_Covariance_Matrix),
        to_trace_two = multiply_matrixes(SONG_Y_Inverse_MFCC_Covariance_Matrix, SONG_X_MFCC_Covariance_Matrix),
        
        to_trace_three_temp = multiply_matrixes(X_MINUS_Y_MFCC_Means, transposed_X_MINUS_Y_MFCC_Means),
        to_trace_three = multiply_matrix_by_scalar(X_SUM_Y_Inverse_MFCC_Covariance_matrix, to_trace_three_temp[0][0]/*one value array actually*/);
    
    return trace_matrix(to_trace_one) + trace_matrix(to_trace_two) + trace_matrix(to_trace_three) - SONG_X_MFCC_Covariance_Matrix.size()*2;
}

//10 times faster than slow version
float get_song_distance(
    vector< vector<float> > &A_array/*13x13*/, vector< vector<float> > &C_array/*13x13*/, vector<float> &E_array/*13*/,
    vector< vector<float> > &B_array/*13x13*/, vector< vector<float> > &D_array/*13x13*/, vector<float> &F_array/*13*/
)
{
    size_t size = A_array.size();//it's 13 always
    
    float tmp = 0, tmp_2 = 0, result = 0;
    
    for(size_t i = 0; i < size; i++)
    {
        tmp_2 = E_array[i] - F_array[i];
        tmp += tmp_2 * tmp_2;
    }
    
    for(size_t i = 0; i < size; i++)
    {
        result += tmp * (C_array[i][i] + D_array[i][i]);
        
        for(size_t k = 0; k < size; k++)
            result += (C_array[i][k] * B_array[k][i]) + (D_array[i][k] * A_array[k][i]);
    }
    
    return result - size*2;
}

void calculate_required_song_features(string &filename, vector< vector<Real> > &MFCC_Covariance_Matrix/*13 x 13 in size*/, vector< vector<Real> > &Inverse_MFCC_Covariance_Matrix/*13 x 13 in size*/, vector<Real> &MFCC_Means/*13 in size*/, bool &is_valid)
{
    cout << "Indexing: " << filename << "\n";
    //all in all, for a single song we need 351 floats = 1,404 bytes to store these extracted features
    //we can easily calculate the inverse matrix so we can avoid storing this one. this way we will save 676 bytes
    //so all in all we only need 728 bytes, asuming the size of these arrays will ALWAYS be the same for any song, so we don't need to waste space for seperators
    
    int sampleRate = 44100;
    int frameSize = 2048;
    int hopSize = 1024;
    
    // we want to compute the MFCC of a file: we need the create the following:
    // audioloader -> framecutter -> windowing -> FFT -> MFCC
    
    AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
    
    Algorithm* audio = NULL;
    
    try
    {
        audio = factory.create("MonoLoader", "filename", filename, "sampleRate", sampleRate);
    }
    catch(essentia::EssentiaException e)
    {
        if(strcmp(e.what(), "AudioLoader ERROR: found 0 streams in the file, expecting only one audio stream") == 0)
        {
            cout << "Skipping because we found 0 audio streams\n";
            delete audio;
            is_valid = false;
            return;
        }
    }
    
    Algorithm* fc = factory.create("FrameCutter", "frameSize", frameSize, "hopSize", hopSize);
    Algorithm* w = factory.create("Windowing", "type", "blackmanharris62");
    Algorithm* spec = factory.create("Spectrum");
    Algorithm* mfcc = factory.create("MFCC");
    
    // Audio -> FrameCutter
    vector<Real> audioBuffer;
    
    audio->output("audio").set(audioBuffer);
    fc->input("signal").set(audioBuffer);
    
    // FrameCutter -> Windowing -> Spectrum
    vector<Real> frame, windowedFrame;
    
    fc->output("frame").set(frame);
    w->input("frame").set(frame);
    
    w->output("frame").set(windowedFrame);
    spec->input("frame").set(windowedFrame);
    
    // Spectrum -> MFCC
    vector<Real> spectrum, mfccCoeffs, mfccBands;
    
    spec->output("spectrum").set(spectrum);
    mfcc->input("spectrum").set(spectrum);
    
    mfcc->output("bands").set(mfccBands);
    mfcc->output("mfcc").set(mfccCoeffs);
    
    audio->compute();
    
    vector< vector<Real> > matrix_mfcc_values;
    
    while (true)
    {
        // compute a frame
        fc->compute();
        
        // if it was the last one (ie: it was empty), then we're done.
        if (!frame.size()) break;
        
        // if the frame is silent, just drop it and go on processing
        if (isSilent(frame)) continue;
        
        w->compute();
        spec->compute();
        mfcc->compute();
        
        matrix_mfcc_values.push_back(mfccCoeffs);
    }
    
    Algorithm* sg = NULL;
    
    if(matrix_mfcc_values.size() > 0)
    {
        Array2D<Real> temp_MFCC_Covariance_Matrix, temp_Inverse_MFCC_Covariance_Matrix, temp_input;
        temp_input = vecvecToArray2D(matrix_mfcc_values);

        sg = factory.create("SingleGaussian");
        
        try
        {
            sg->input("matrix").set(temp_input);
            sg->output("mean").set(MFCC_Means);
            sg->output("covariance").set(temp_MFCC_Covariance_Matrix);
            sg->output("inverseCovariance").set(temp_Inverse_MFCC_Covariance_Matrix);
            sg->compute();
            
            MFCC_Covariance_Matrix = array2DToVecvec(temp_MFCC_Covariance_Matrix);
            Inverse_MFCC_Covariance_Matrix = array2DToVecvec(temp_Inverse_MFCC_Covariance_Matrix);
            is_valid = true;
        }
        catch (essentia::EssentiaException e)
        {
            is_valid = false;
            
            if(strcmp(e.what(), "SingleGaussian: Cannot solve linear system because matrix is singular") == 0)
                cout << "Skipping because matrix is singular\n";
            else
            {
                cout << e.what() << "\n";
                exit(1);
            }
        }
    }
    else
        is_valid = false;
    
    delete audio;
    delete fc;
    delete w;
    delete spec;
    delete mfcc;
    
    if(sg != NULL)
        delete sg;
}

class song_item
{
private:
    string song_hdd_path;
    vector< vector<Real> > MFCC_Covariance_Matrix, Inverse_MFCC_Covariance_Matrix;
    vector<Real> MFCC_Means;
    
    void do_load_data(float* song_data)
    {
        size_t pos = 0;

        for(size_t i = 0; i < 13; i++)
            for(size_t j = 0; j < 13; j++)
                MFCC_Covariance_Matrix[i][j] = song_data[pos++];

        for(size_t i = 0; i < 13; i++)
            for(size_t j = 0; j < 13; j++)
                Inverse_MFCC_Covariance_Matrix[i][j] = song_data[pos++];

        for(size_t i = 0; i < 13; i++)
            MFCC_Means[i] = song_data[pos++];
    }
    
    bool load_from_disk(string &song_hdd_path)
    {
        size_t total_size = 13*13+13*13+13;
        float* song_data = new float[total_size];
        
        if(!read_song_data(song_hdd_path, song_data, total_size))
        {
            delete []song_data;
            return false;
        }
        else
        {
            do_load_data(song_data);
            delete []song_data;
            return true;
        }
    }
    
    bool generate_and_write_to_disk(string &song_hdd_path)
    {
        bool is_valid = true;
        calculate_required_song_features(song_hdd_path, MFCC_Covariance_Matrix, Inverse_MFCC_Covariance_Matrix, MFCC_Means, is_valid);
        
        if(!is_valid) return false;
        
        size_t total_size = 13*13+13*13+13;
        float* song_data = new float[total_size];
        
        size_t pos = 0;
        
        for(size_t i = 0; i < 13; i++)
            for(size_t j = 0; j < 13; j++)
                song_data[pos++] = MFCC_Covariance_Matrix[i][j];
        
        for(size_t i = 0; i < 13; i++)
            for(size_t j = 0; j < 13; j++)
                song_data[pos++] = Inverse_MFCC_Covariance_Matrix[i][j];
        
        for(size_t i = 0; i < 13; i++)
            song_data[pos++] = MFCC_Means[i];
        
        store_song_data(song_hdd_path, song_data, total_size);
        
        delete []song_data;
        
        return true;
    }
    
public:
    song_item(){}
    
    song_item(string &song_full_path, bool &is_valid) : MFCC_Covariance_Matrix(13, std::vector<Real>(13)), Inverse_MFCC_Covariance_Matrix(13, std::vector<Real>(13)), MFCC_Means(13)
    {
        is_valid = true;
        song_hdd_path = song_full_path;
        
        if(!load_from_disk(song_hdd_path))
            if(!generate_and_write_to_disk(song_hdd_path))
                is_valid = false;
    }
    
    song_item(string song_full_path, float* data_array) : MFCC_Covariance_Matrix(13, std::vector<Real>(13)), Inverse_MFCC_Covariance_Matrix(13, std::vector<Real>(13)), MFCC_Means(13)
    {
        song_hdd_path = song_full_path;
        do_load_data(data_array);
    }
    
    float compare(song_item &other)
    {
        return get_song_distance(
            MFCC_Covariance_Matrix, Inverse_MFCC_Covariance_Matrix, MFCC_Means,
            other.MFCC_Covariance_Matrix, other.Inverse_MFCC_Covariance_Matrix, other.MFCC_Means
        );
    }
    
    string path()
    {
        return song_hdd_path;
    }
    
    string name()
    {
        int found = song_hdd_path.find_last_of(seperator);
        
        if(found > 0)
            return song_hdd_path.substr(found+1);
        else
            return song_hdd_path;
    }
};

vector<string> get_music_files_in_directory(string &directory)
{
    vector<string> output;
    vector<string> accepted_types = {".mp3", ".aac", ".flac", ".ogg", ".m4a", ".wma", ".wav"};
    
    DIR *dir;
    struct dirent *ent;
    
    if ((dir = opendir (directory.c_str())) != NULL)
    {
        while ((ent = readdir (dir)) != NULL)
        {
            string file_now = ent->d_name;
            
            if(file_now != "." && file_now != "..")
            {
                string full_file_name = directory+(string(1, directory.back()) == seperator ? "" : seperator)+file_now;
                
                struct stat fstat;
                
                if (stat(full_file_name.c_str(), &fstat) >= 0 && S_ISDIR(fstat.st_mode))//valid and dir
                {
                    vector<string> result_files = get_music_files_in_directory(full_file_name);
                    output.insert(std::end(output), std::begin(result_files), std::end(result_files));
                }
                else
                {
                    int found = file_now.find_last_of(".");
                    
                    if(found > 0)
                    {
                        string ext = file_now.substr(found);

                        if(std::find(accepted_types.begin(), accepted_types.end(), ext) != accepted_types.end())
                            output.push_back(full_file_name);
                    }
                }
            }
        }
        
        closedir (dir);
    }
    
    return output;
}

vector<song_item> load_songs(vector<string> path_list)
{
    vector<song_item> songs;
    
    for(size_t i = 0; i < path_list.size(); i++)
    {
        vector<string> song_names = get_music_files_in_directory(path_list[i]);

        for(size_t j = 0; j < song_names.size(); j++)
        {
            bool is_valid = true;
            song_item tmp = song_item(song_names[j], is_valid);
            
            if(is_valid)
                songs.push_back(tmp);
            else
                cout << "Skipping invalid: " << song_names[j] << "\n";
        }
    }
    
    return songs;
}

vector<song_item> reall_all_songs_from_db()
{
    vector<song_item> results;
    
    sqlite3_stmt *stmt = NULL;
    
    if (sqlite3_prepare(db, "SELECT path, data FROM music;", -1, &stmt, NULL))
    {
        cerr << "SQL ERROR WHEN READING: " << sqlite3_errmsg(db) << endl;
        exit(1);
    }
    
    while (true)
    {
        int res = sqlite3_step (stmt);
        
        if (res == SQLITE_ROW)
        {
            const char *text  = (const char*)sqlite3_column_text(stmt, 0);
            float *mem_got = (float*)sqlite3_column_blob(stmt, 1);
            results.push_back(song_item(string(text), mem_got));
        }
        else if (res == SQLITE_DONE)
            break;
        else
        {
            cerr << "SQL ERROR WHEN READING: " << sqlite3_errmsg(db) << endl;
            exit(1);
        }
    }
    
    sqlite3_finalize(stmt);
    return results;
}

class comparison_result
{
private:
    float song_distance;
    song_item *song;
    
public:
    comparison_result(){}
        
    comparison_result(song_item &song, float song_distance)
    {
        this->song = &song;
        this->song_distance = song_distance;
    }
    
    float distance()
    {
        return song_distance;
    }
    
    string path()
    {
        return song->path();
    }
    
    string name()
    {
        return song->name();
    }
    
    bool operator<(const comparison_result &other) const
    {
        return song_distance < other.song_distance;
    }
};

vector<comparison_result> get_similar_songs(vector<song_item> &songs, song_item &song)
{
    vector<comparison_result> results(songs.size());
    
    for(size_t i = 0; i < songs.size(); i++)
        results[i] = comparison_result(songs[i], song.compare(songs[i]));
    
    std::sort(results.begin(), results.end(), less<comparison_result>());
    
    return results;
}

void start_do_search(string input_file)
{
    vector<song_item> songs = reall_all_songs_from_db();
    
    bool valid = false;
    song_item tmp = song_item(input_file, valid);
    
    if(!valid)
    {
        cout << "Can't search because input song not in our database.\n";
        return;
    }
    else
    {
        auto start = chrono::steady_clock::now();
        
        vector<comparison_result> similar_songs = get_similar_songs(songs, tmp);
        
        auto time_search_took = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
        
        std::ofstream ofs ("output.txt", std::ofstream::out);
        ofs << "\nTop 1000 Similar Songs to " << tmp.name() << "\n\n";
        
        //exclude songs with distance <= 1
        for(size_t i = 0, got = 0; i < similar_songs.size() && got < 1000; i++)
        {
            float distance = similar_songs[i].distance();
            
            if(distance > 1)
            {
                ofs << "Distance " << distance << " Disk Path: " << similar_songs[i].path() << "\n";
                got++;
            }
        }
        
        ofs.close();
        cout << "Stored search results in file output.txt. Time search took: " << time_search_took << "ms\n";
    }
}

int main(int argc, char* argv[])
{
    //we can optimize indexing by using multiple threads, if essentia supports it
    
    //some very silent songs always appear to have distance 0 from every song. probably mp3 encoding error
    //also maybe normalize sound volume (i don't think essentia does this by default)
    
    essentia::init();//start essentia
    load_db();
    
    vector<string> args;//excluding program name
    
    for (int i = 1; i < argc; i++)
        args.push_back(argv[i]);
    
    if(args.size() == 2 && args[0] == "-search")
        start_do_search(args[1]);
    else if(args.size() == 1 && args[0] == "-index")
    {
        vector<song_item> songs = load_songs({
            "/home/music_to_index",
        });
        
        cout << "\nIndexing Finished\n";
    }
    else
    {
        cout << "Usage 1: -index\nUsage 2: -search input_filename\n";
        close_db();
        essentia::shutdown();//close essentia
        return 0;
    }
    
    close_db();
    essentia::shutdown();//close essentia
    return 0;
}
