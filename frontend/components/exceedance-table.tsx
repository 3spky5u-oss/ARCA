"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatFactor, cn } from "@/lib/utils";
import { ArrowUpDown, ArrowDown, ArrowUp, AlertTriangle } from "lucide-react";
import type { Exceedance } from "@/lib/api";

interface ExceedanceTableProps {
  exceedances: Exceedance[];
}

type SortField = "exceedance_factor" | "parameter" | "sample_id" | "value";
type SortDirection = "asc" | "desc";

function renderSortIcon(field: SortField, sortField: SortField, sortDirection: SortDirection) {
  if (field !== sortField) {
    return <ArrowUpDown className="h-4 w-4 ml-1 opacity-50" />;
  }

  return sortDirection === "desc" ? (
    <ArrowDown className="h-4 w-4 ml-1" />
  ) : (
    <ArrowUp className="h-4 w-4 ml-1" />
  );
}

export function ExceedanceTable({ exceedances }: ExceedanceTableProps) {
  const [sortField, setSortField] = useState<SortField>("exceedance_factor");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [filterParameter, setFilterParameter] = useState<string>("all");

  // Get unique parameters for filter
  const uniqueParameters = useMemo(() => {
    const params = new Set(exceedances.map((e) => e.parameter));
    return Array.from(params).sort();
  }, [exceedances]);

  // Filter and sort exceedances
  const sortedExceedances = useMemo(() => {
    let filtered = exceedances;

    if (filterParameter !== "all") {
      filtered = exceedances.filter((e) => e.parameter === filterParameter);
    }

    return [...filtered].sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case "exceedance_factor":
          comparison = a.exceedance_factor - b.exceedance_factor;
          break;
        case "parameter":
          comparison = a.parameter.localeCompare(b.parameter);
          break;
        case "sample_id":
          comparison = a.sample_id.localeCompare(b.sample_id);
          break;
        case "value":
          comparison = a.value - b.value;
          break;
      }

      return sortDirection === "desc" ? -comparison : comparison;
    });
  }, [exceedances, sortField, sortDirection, filterParameter]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(sortDirection === "desc" ? "asc" : "desc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  if (exceedances.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 bg-green-100 dark:bg-green-900/30 rounded-full">
              <AlertTriangle className="h-8 w-8 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-lg font-medium text-green-700 dark:text-green-300">
                No Exceedances Found
              </p>
              <p className="text-sm text-muted-foreground">
                All parameters are within guidelines
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            Exceedances ({sortedExceedances.length})
          </CardTitle>
          <Select value={filterParameter} onValueChange={setFilterParameter}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Filter by parameter" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Parameters</SelectItem>
              {uniqueParameters.map((param) => (
                <SelectItem key={param} value={param}>
                  {param}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-3 px-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleSort("sample_id")}
                    className="font-semibold -ml-3"
                  >
                    Sample
                    {renderSortIcon("sample_id", sortField, sortDirection)}
                  </Button>
                </th>
                <th className="text-left py-3 px-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleSort("parameter")}
                    className="font-semibold -ml-3"
                  >
                    Parameter
                    {renderSortIcon("parameter", sortField, sortDirection)}
                  </Button>
                </th>
                <th className="text-right py-3 px-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleSort("value")}
                    className="font-semibold -mr-3"
                  >
                    Result
                    {renderSortIcon("value", sortField, sortDirection)}
                  </Button>
                </th>
                <th className="text-right py-3 px-4">Guideline</th>
                <th className="text-right py-3 px-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleSort("exceedance_factor")}
                    className="font-semibold -mr-3"
                  >
                    Factor
                    {renderSortIcon("exceedance_factor", sortField, sortDirection)}
                  </Button>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedExceedances.map((exc, i) => (
                <tr
                  key={`${exc.sample_id}-${exc.parameter}-${i}`}
                  className={cn(
                    "border-b last:border-0 transition-colors",
                    exc.exceedance_factor >= 2
                      ? "bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/30"
                      : "hover:bg-muted/50"
                  )}
                >
                  <td className="py-3 px-4 font-mono text-sm">{exc.sample_id}</td>
                  <td className="py-3 px-4">{exc.parameter}</td>
                  <td className="py-3 px-4 text-right font-mono">
                    <span className="text-red-600 dark:text-red-400 font-medium">
                      {exc.value_str}
                    </span>
                    <span className="text-muted-foreground text-sm ml-1">
                      {exc.unit}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-muted-foreground">
                    {exc.guideline}
                    <span className="text-sm ml-1">{exc.guideline_unit}</span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span
                      className={cn(
                        "font-bold px-2 py-1 rounded-full",
                        exc.exceedance_factor >= 5
                          ? "bg-red-600 text-white"
                          : exc.exceedance_factor >= 2
                          ? "bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300"
                          : "bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300"
                      )}
                    >
                      {formatFactor(exc.exceedance_factor)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
