/*
 * Implementation of affnity schedule and apply it to run on loop1 and loop2.
 * Author: B177252
 */
#include <stdio.h>
#include <math.h>

#define N 1729
#define reps 1000
#include <omp.h>

double a[N][N], b[N][N], c[N];
int jmax[N];

void init1(void);
void init2(void);
void runloop(int);
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);
int find_most_load_thread(int *, int *, int);
int get_next_iterations(int *, int *, int, int, int *, int *);

int main(int argc, char *argv[])
{

  double start1, start2, end1, end2;
  int r;

  init1();

  start1 = omp_get_wtime();

  for (r = 0; r < reps; r++)
  {
    runloop(1);
  }

  end1 = omp_get_wtime();

  valid1();

  printf("Total time for %d reps of loop 1 = %f\n", reps, (float)(end1 - start1));

  init2();

  start2 = omp_get_wtime();

  for (r = 0; r < reps; r++)
  {
    runloop(2);
  }

  end2 = omp_get_wtime();

  valid2();

  printf("Total time for %d reps of loop 2 = %f\n", reps, (float)(end2 - start2));
}

void init1(void)
{
  int i, j;

  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      a[i][j] = 0.0;
      b[i][j] = 1.618 * (i + j);
    }
  }
}

void init2(void)
{
  int i, j, expr;

  for (i = 0; i < N; i++)
  {
    expr = i % (4 * (i / 60) + 1);
    if (expr == 0)
    {
      jmax[i] = N / 2;
    }
    else
    {
      jmax[i] = 1;
    }
    c[i] = 0.0;
  }

  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      b[i][j] = (double)(i * j + 1) / (double)(N * N);
    }
  }
}

void runloop(int loopid)
{

  /*
   * Number of threads.
   */
  int nthreads = omp_get_max_threads();

  /*                                                                                                                                                                                                       
   * Starting and ending index of remaining local set.                                                                                                                                                     
   */
  int remaining_start[nthreads];
  int remaining_end[nthreads];

/*
   * Parallel section.
   */
#pragma omp parallel default(none) shared(loopid, nthreads, remaining_start, remaining_end)
  {

    /*
     * ID of thread.
     */
    int myid = omp_get_thread_num();

    /*
     * Size of local set.
     */
    int ipt = (int)ceil((double)N / (double)nthreads);

    /*
     * Starting and ending index of local set.
     */
    int lo = myid * ipt;
    int hi = (myid + 1) * ipt;

    /*
     * Make sure the index does not exceed maximum number.
     */
    if (hi > N)
      hi = N;

    /*
     * Starting and ending index of iterations to be executed.
     */
    int low = 0;
    int high = 0;

    /*
     * Initialize remaining local set for each thread.
     */
    remaining_start[myid] = lo;
    remaining_end[myid] = hi;

    /*
     * Flag is used to determine when to stop. 
     */
    int flag = 0;

/*
     * Add barrier to avoid race condition.
     */
#pragma omp barrier
    do
    {

      /*
        * Check which loop to execute.
        */
      switch (loopid)
      {
      case 1:
        loop1chunk(low, high);
        break;
      case 2:
        loop2chunk(low, high);
        break;
      }

/*
       * Critical section used to avoid race condition.
       */
#pragma omp critical
      {

        /*
        * Get the next iterations to execute.
        */
        flag = get_next_iterations(remaining_start, remaining_end, nthreads, myid, &low, &high);
      }
    } while (flag != -1);
  }
}

int get_next_iterations(int *remaining_start, int *remaining_end, int nthreads, int myid, int *low, int *high)
{

  /*
    * ID of either current thread or most loaded thread.
    */
  int index;

  /*
    * If a thread has not finished itself, index is its own ID, otherwise index is the ID of the most loaded thread.
    */
  if (remaining_end[myid] - remaining_start[myid] > 0)
  {
    index = myid;
  }
  else
  {
    index = find_most_load_thread(remaining_start, remaining_end, nthreads);
  }

  /*
    * If index is -1, it means all the iterations are finished, the return -1.
    */
  if (index == -1)
  {
    return -1;
  }

  /*
    * Calculate the number of iterations to execute next turn.
    */
  int remaining = remaining_end[index] - remaining_start[index];
  int chunksize = (int)ceil(remaining * 1.0 / nthreads);

  /*
    * Update the starting and ending index of iterations to be executed.
    */
  *low = remaining_start[index];
  *high = remaining_start[index] + chunksize;

  /*
    * Update the starting index of the thread's local set.
    */
  remaining_start[index] += chunksize;

  return 0;
}

int find_most_load_thread(int *remaining_start, int *remaining_end, int nthreads)
{

  /*
   * Index of most loaded thread.
   */
  int index = -1;

  /*
   * Maximum number of remaining.
   */
  int max_remaining = 0;

  /*
   * Find the index of most loaded thread. 
   */
  for (int i = 0; i < nthreads; i++)
  {
    int remaining = remaining_end[i] - remaining_start[i];
    if (remaining > max_remaining)
    {
      index = i;
      max_remaining = remaining;
    }
  }
  return index;
}

void loop1chunk(int lo, int hi)
{
  int i, j;

  for (i = lo; i < hi; i++)
  {
    for (j = N - 1; j > i; j--)
    {
      a[i][j] += cos(b[i][j]);
    }
  }
}

void loop2chunk(int lo, int hi)
{
  int i, j, k;
  double rN2;

  rN2 = 1.0 / (double)(N * N);

  for (i = lo; i < hi; i++)
  {
    for (j = 0; j < jmax[i]; j++)
    {
      for (k = 0; k < j; k++)
      {
        c[i] += (k + 1) * log(b[i][j]) * rN2;
      }
    }
  }
}

void valid1(void)
{
  int i, j;
  double suma;

  suma = 0.0;
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      suma += a[i][j];
    }
  }
  printf("Loop 1 check: Sum of a is %lf\n", suma);
}

void valid2(void)
{
  int i;
  double sumc;

  sumc = 0.0;
  for (i = 0; i < N; i++)
  {
    sumc += c[i];
  }
  printf("Loop 2 check: Sum of c is %f\n", sumc);
}
