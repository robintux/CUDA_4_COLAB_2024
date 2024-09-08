#pragma once

template<class my_float_t>
void check_result(const int matrix_size, const my_float_t* host_matrix) {
  printf("Result:\n");
  long long int errors = 0;
  for (long long int i = 0;i < matrix_size;i++)
    {
      for (long long int j = 0;j < matrix_size;j++)
  	{
          const my_float_t expected_result = (my_float_t) i * ((my_float_t)matrix_size * j + (my_float_t) (2 + j) * matrix_size * (matrix_size - 1) / 2 + (my_float_t) 2 * matrix_size * (matrix_size - 1) * (2 * matrix_size - 1) / 6);
          if (host_matrix[i * matrix_size + j] != expected_result ) 
            {
              errors++;
              if (errors <= 10) 
                printf("\nError in row %d col %d - Expected: %2.0f, Found: %2.0f\n", (int) i, (int) j, expected_result, host_matrix[i * matrix_size + j]);
            }
          if (j >= 8 || i >= 32) continue;
          printf("%2.0f\t\t", host_matrix[i * matrix_size + j]);
        }
      if (i >= 32) continue;
      printf("\n");
    }
  printf("\n");
  
  if (errors == 0)
    {
      printf("VERIFICATION PASSED\n");
    }
  else
    {
      printf("VERIFICATION FAILED\n");
    }
}
