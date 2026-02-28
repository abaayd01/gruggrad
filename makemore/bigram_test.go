package makemore

import (
	"fmt"
	"slices"
	"testing"
)

func TestRuneToIdx(t *testing.T) {
	testCases := [](struct {
		input  rune
		output int
	}){
		{
			input:  'a',
			output: 0,
		},
		{
			input:  'e',
			output: 4,
		},
		{
			input:  'z',
			output: 25,
		},
		{
			input:  BoundaryRune,
			output: BoundaryIdx,
		},
	}

	for _, tc := range testCases {
		result := RuneToIdx(tc.input)
		if result != tc.output {
			t.Errorf("expected: %d, but got %d for input: %c", tc.output, result, tc.input)
		}
	}
}

func TestToRunePairs(t *testing.T) {
	testCases := [](struct {
		input  string
		output [][2]rune
	}){
		{
			input: frameString("a"),
			output: [][2]rune{
				{BoundaryRune, 'a'},
				{'a', BoundaryRune},
			},
		},
		{
			input: frameString("abcd"),
			output: [][2]rune{
				{BoundaryRune, 'a'},
				{'a', 'b'},
				{'b', 'c'},
				{'c', 'd'},
				{'d', BoundaryRune},
			},
		},
	}

	for _, tc := range testCases {
		result := convertToRunePairs(tc.input)
		if !slices.Equal(result, tc.output) {
			t.Errorf("expected: %v, but got %v for input: %s", tc.output, result, tc.input)
		}
	}

}

func TestGenerateProbabilityMatrix(t *testing.T) {
	t.Run("single word", func(t *testing.T) {
		result := GenerateProbabilityMatrix([]string{"aabc"})

		// For "aabc", the transitions are: (.→a), (a→a), (a→b), (b→c), (c→.)
		// Rows are normalized independently to represent P(next|current).

		testCases := []struct {
			name     string
			from     rune
			to       rune
			expected float64
		}{
			{".→a", BoundaryRune, 'a', 1.0},
			{"a→a", 'a', 'a', 0.5},
			{"a→b", 'a', 'b', 0.5},
			{"b→c", 'b', 'c', 1.0},
			{"c→.", 'c', BoundaryRune, 1.0},
			{"a→. (doesn't exist)", 'a', BoundaryRune, 0.00},
			{".→b (doesn't exist)", BoundaryRune, 'b', 0.00},
		}

		for _, tc := range testCases {
			prob := result.Get(RuneToIdx(tc.from), RuneToIdx(tc.to))
			if fmt.Sprintf("%.2f", prob) != fmt.Sprintf("%.2f", tc.expected) {
				t.Errorf("%s: expected %.2f, but got %.2f", tc.name, tc.expected, prob)
			}
		}
	})

	t.Run("multiple words with repeated bigrams", func(t *testing.T) {
		result := GenerateProbabilityMatrix([]string{"ab", "ab"})

		// For "ab" (twice), row-normalized conditionals are unchanged by repeats.

		testCases := []struct {
			name     string
			from     rune
			to       rune
			expected float64
		}{
			{".→a", BoundaryRune, 'a', 1.0},
			{"a→b", 'a', 'b', 1.0},
			{"b→.", 'b', BoundaryRune, 1.0},
			{"a→a (doesn't exist)", 'a', 'a', 0.00},
		}

		for _, tc := range testCases {
			prob := result.Get(RuneToIdx(tc.from), RuneToIdx(tc.to))
			if fmt.Sprintf("%.4f", prob) != fmt.Sprintf("%.4f", tc.expected) {
				t.Errorf("%s: expected %.4f, but got %.4f", tc.name, tc.expected, prob)
			}
		}
	})

	t.Run("empty strings are skipped", func(t *testing.T) {
		result := GenerateProbabilityMatrix([]string{"a", "", "b"})

		// "a": (.→a), (a→.)
		// "": skipped
		// "b": (.→b), (b→.)

		testCases := []struct {
			name     string
			from     rune
			to       rune
			expected float64
		}{
			{".→a", BoundaryRune, 'a', 0.5},
			{"a→.", 'a', BoundaryRune, 1.0},
			{".→b", BoundaryRune, 'b', 0.5},
			{"b→.", 'b', BoundaryRune, 1.0},
		}

		for _, tc := range testCases {
			prob := result.Get(RuneToIdx(tc.from), RuneToIdx(tc.to))
			if fmt.Sprintf("%.4f", prob) != fmt.Sprintf("%.4f", tc.expected) {
				t.Errorf("%s: expected %.4f, but got %.4f", tc.name, tc.expected, prob)
			}
		}
	})
}

func TestBuildTrainingExamples(t *testing.T) {
	examples := buildTrainingExamples("./names.txt")
	fmt.Printf("Input:\n%+v\n\n", examples[0].Input)
	fmt.Printf("Output:\n%+v\n\n", examples[0].Output)
}

func TestConvertToNGramString(t *testing.T) {
	input := "emma"
	n := 3

	// expectedResult := []string{
	// 	string(BoundaryRune) + string(BoundaryRune) + "e",
	// 	string(BoundaryRune) + "e" + "m",
	// 	"emm",
	// 	"mma",
	// 	"ma" + string(BoundaryRune),
	// 	"a" + string(BoundaryRune) + string(BoundaryRune),
	// }

	result := ConvertToNGramString(input, n)

	t.Logf("\nresult: %+x\n", result)
}
